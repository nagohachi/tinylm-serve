/// <reference lib="webworker" />
import { requestDevice } from "./device.ts";
import { TinyLM } from "./model.ts";
import type { GenerationStats, WorkerRequest, WorkerResponse } from "./protocol.ts";
import { type SamplingParams, sampleToken } from "./sampler.ts";
import { type ChatMessage, UnigramTokenizer, renderChatTemplate } from "./tokenizer.ts";
import { HttpChunkSource } from "./weights.ts";

const CACHE_NAME = "tinylm-weights-v1";
const FILES_CACHE = "tinylm-files-v1";
const HUB = "https://huggingface.co";
const KEY_ORIGIN = "https://tinylm-cache.invalid";
const STOP_SEQUENCES = ["\n\n### 指示:", "\n\n### 応答:"];

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: WorkerResponse) => ctx.postMessage(m);

let device: GPUDevice | null = null;
let adapterInfo = "";
let model: TinyLM | null = null;
let tokenizer: UnigramTokenizer | null = null;
let loadedId = "";
let stopRequested = false;
/** First uncaptured GPU error since the last check. */
let gpuError: string | null = null;
let busy: Promise<unknown> = Promise.resolve();

async function getDevice(): Promise<GPUDevice> {
  if (device) return device;
  if (!("gpu" in navigator)) throw new Error("このブラウザは WebGPU に対応していません");
  const r = await requestDevice(navigator.gpu);
  const dev = r.device;
  device = dev;
  adapterInfo = r.adapterInfo;
  dev.addEventListener("uncapturederror", (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error("WebGPU error:", msg);
    gpuError ??= msg;
  });
  dev.lost.then((info) => {
    if (device !== dev) return;
    device = null;
    model = null;
    loadedId = "";
    post({ type: "loadError", message: `GPU デバイスが失われました: ${info.message}` });
  });
  return dev;
}

function takeGpuError(): string | null {
  const e = gpuError;
  gpuError = null;
  return e;
}

async function openCache(name: string): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(name);
  } catch {
    return null;
  }
}

/** What to download: the commit to read small files from, and the weight file's size and content hash. */
interface Manifest {
  sha: string;
  size: number;
  hash: string;
}

const manifestKey = (repo: string) => `${KEY_ORIGIN}/${repo}/manifest.json`;
const weightsKeyBase = (repo: string, hash: string) => `${KEY_ORIGIN}/${repo}/weights/${hash}`;

async function readManifest(repo: string): Promise<Manifest | null> {
  try {
    const hit = await (await openCache(FILES_CACHE))?.match(manifestKey(repo));
    return hit ? ((await hit.json()) as Manifest) : null;
  } catch {
    return null;
  }
}

async function resolveManifest(repo: string, fallbackBytes: number): Promise<Manifest> {
  try {
    const res = await fetch(`${HUB}/api/models/${repo}?blobs=true`);
    if (!res.ok) throw new Error(String(res.status));
    const info = (await res.json()) as {
      sha: string;
      siblings: { rfilename: string; size?: number; lfs?: { sha256: string; size: number } }[];
    };
    const st = info.siblings.find((s) => s.rfilename === "model.safetensors");
    return {
      sha: info.sha,
      size: st?.lfs?.size ?? st?.size ?? fallbackBytes,
      // The LFS hash identifies the weight file itself, so commits that only touch
      // other files (README etc.) keep the cached weights valid.
      hash: st?.lfs?.sha256 ?? info.sha,
    };
  } catch {
    // Offline or API failure: use what was loaded last time.
    const m = await readManifest(repo);
    if (m) return m;
    throw new Error("Hugging Face Hub に接続できませんでした");
  }
}

async function fetchCached(repo: string, sha: string, file: string): Promise<Response> {
  const key = `${KEY_ORIGIN}/${repo}/${sha}/${file}`;
  const cache = await openCache(FILES_CACHE);
  try {
    const hit = await cache?.match(key);
    if (hit) return hit;
  } catch {
    // fall through to the network
  }
  const res = await fetch(`${HUB}/${repo}/resolve/${sha}/${file}`);
  if (!res.ok) throw new Error(`${file} の取得に失敗しました (HTTP ${res.status})`);
  try {
    await cache?.put(key, res.clone());
  } catch {
    // ignore quota errors
  }
  return res;
}

/** After a successful load: remember the manifest and drop files of other revisions. */
async function commitRevision(repo: string, m: Manifest) {
  const files = await openCache(FILES_CACHE);
  const weights = await openCache(CACHE_NAME);
  try {
    await files?.put(manifestKey(repo), new Response(JSON.stringify(m), { headers: { "Content-Type": "application/json" } }));
    const prefix = `${KEY_ORIGIN}/${repo}/`;
    for (const k of (await files?.keys()) ?? []) {
      if (k.url.startsWith(prefix) && k.url !== manifestKey(repo) && !k.url.startsWith(`${prefix}${m.sha}/`)) {
        await files!.delete(k);
      }
    }
    const keep = weightsKeyBase(repo, m.hash) + "/";
    for (const k of (await weights?.keys()) ?? []) {
      if (k.url.startsWith(prefix) && !k.url.startsWith(keep)) await weights!.delete(k);
    }
  } catch {
    // best effort
  }
}

async function load(modelId: string, repo: string, fallbackBytes: number) {
  if (loadedId === modelId && model) {
    post({ type: "ready", modelId, adapter: adapterInfo, contextLength: model.cfg.maxSeq });
    return;
  }
  model?.destroy();
  model = null;
  tokenizer = null;
  loadedId = "";
  post({ type: "status", text: "WebGPU を初期化しています" });
  const dev = await getDevice();
  takeGpuError();
  post({ type: "status", text: "モデル情報を取得しています" });
  const manifest = await resolveManifest(repo, fallbackBytes);
  const [config, tokJson] = await Promise.all([
    fetchCached(repo, manifest.sha, "config.json").then((r) => r.json()),
    fetchCached(repo, manifest.sha, "tokenizer.json").then((r) => r.json()),
  ]);
  const tok = new UnigramTokenizer(tokJson);
  const src = await HttpChunkSource.open(
    `${HUB}/${repo}/resolve/${manifest.sha}/model.safetensors`,
    manifest.size,
    CACHE_NAME,
    weightsKeyBase(repo, manifest.hash),
  );
  const fromCache = (await src.cachedChunks()) === src.chunkCount;
  post({ type: "status", text: fromCache ? "キャッシュから重みを読み込んでいます" : "重みをダウンロードしています" });
  const m = await TinyLM.load(dev, config, src, {
    onProgress: ({ doneBytes, totalBytes }) => post({ type: "progress", doneBytes, totalBytes, fromCache }),
  });
  try {
    post({ type: "status", text: "シェーダを準備しています" });
    // Build and run the decode (1 token) and prefill (8 tokens) programs once, so that
    // shader or limit problems on this GPU surface here instead of as garbage output.
    dev.pushErrorScope("validation");
    await m.prefill(Array(8).fill(tok.bosId));
    await m.step(tok.bosId);
    const err = (await dev.popErrorScope())?.message ?? takeGpuError();
    if (err) throw new Error(`このブラウザ / GPU ではモデルを実行できませんでした: ${err}`);
    m.resetCache();
  } catch (e) {
    m.destroy();
    throw e;
  }
  model = m;
  tokenizer = tok;
  loadedId = modelId;
  await commitRevision(repo, manifest);
  post({ type: "ready", modelId, adapter: adapterInfo, contextLength: m.cfg.maxSeq });
}

/** Encodes the conversation, dropping the oldest turns if it does not fit in the context. */
function encodeConversation(messages: ChatMessage[], reserve: number, maxSeq: number) {
  const tok = tokenizer!;
  const system = messages.filter((m) => m.role === "system");
  let rest = messages.filter((m) => m.role !== "system");
  let dropped = 0;
  for (;;) {
    const ids = tok.encode(renderChatTemplate([...system, ...rest], true));
    if (ids.length + reserve <= maxSeq || rest.length <= 1) return { ids, dropped };
    // Drop the oldest exchange (user + assistant).
    const cut = rest.length > 2 && rest[1].role === "assistant" ? 2 : 1;
    rest = rest.slice(cut);
    dropped += cut;
  }
}

async function generate(id: number, messages: ChatMessage[], params: SamplingParams) {
  if (!model || !tokenizer) throw new Error("モデルが読み込まれていません");
  const m = model;
  const tok = tokenizer;
  takeGpuError();
  const maxSeq = m.cfg.maxSeq;
  const reserve = Math.min(params.maxNewTokens, Math.floor(maxSeq / 2));
  const { ids, dropped } = encodeConversation(messages, reserve, maxSeq);
  if (ids.length >= maxSeq) throw new Error(`入力が長すぎます（${ids.length} トークン、上限 ${maxSeq}）`);
  let t0 = performance.now();
  let logits = await m.prefill(ids, (done, total) => post({ type: "prefill", id, done, total }));
  const prefillErr = takeGpuError();
  if (prefillErr) throw new Error(`GPU エラーが発生しました: ${prefillErr}`);
  const reusedTokens = m.lastReused;
  const prefillMs = performance.now() - t0;
  t0 = performance.now();
  const generated: number[] = [];
  const banned = [0, 1, 3, 4, 5, 6, 7].filter((i) => i !== tok.eosId);
  let emitted = "";
  let stopReason: GenerationStats["stopReason"] = "length";
  let finalText = "";
  for (let i = 0; i < params.maxNewTokens; i++) {
    if (stopRequested) {
      stopReason = "stopped";
      break;
    }
    const next = sampleToken(logits, tok.vocabSize, banned, params, generated);
    if (next === tok.eosId) {
      stopReason = "eos";
      break;
    }
    generated.push(next);
    // Lenient decoding: an incomplete UTF-8 tail becomes one U+FFFD instead of
    // turning every byte of the run (including complete characters) into U+FFFD.
    let text = tok.decode(generated, { lenient: true });
    const stopAt = STOP_SEQUENCES.map((s) => text.indexOf(s)).filter((k) => k >= 0);
    if (stopAt.length) {
      text = text.slice(0, Math.min(...stopAt));
      stopReason = "stop-sequence";
    }
    finalText = text;
    // Hold back text while a multi-byte character or a possible stop sequence is incomplete.
    let safe = text;
    if (safe.endsWith("�")) safe = safe.replace(/�+$/, "");
    if (stopReason !== "stop-sequence") {
      for (const s of STOP_SEQUENCES) {
        for (let k = Math.min(s.length - 1, safe.length); k > 0; k--) {
          if (safe.endsWith(s.slice(0, k))) {
            safe = safe.slice(0, safe.length - k);
            break;
          }
        }
      }
    }
    if (safe.length > emitted.length && safe.startsWith(emitted)) {
      post({ type: "token", id, text: safe.slice(emitted.length) });
      emitted = safe;
    }
    if (stopReason === "stop-sequence") break;
    if (m.position + 1 >= maxSeq) {
      stopReason = "context";
      break;
    }
    logits = await m.step(next);
    const err = takeGpuError();
    if (err) throw new Error(`GPU エラーが発生しました: ${err}`);
  }
  const decodeMs = performance.now() - t0;
  const text = finalText.replace(/�+$/, "");
  post({
    type: "done",
    id,
    text,
    stats: {
      promptTokens: ids.length,
      reusedTokens,
      newTokens: generated.length,
      prefillMs,
      decodeMs,
      stopReason,
      droppedMessages: dropped,
    },
  });
}

async function cacheStatus(repos: string[]) {
  const cached: Record<string, { bytes: number; complete: boolean }> = {};
  const cache = await openCache(CACHE_NAME);
  try {
    const keys = (await cache?.keys()) ?? [];
    for (const repo of repos) {
      const manifest = await readManifest(repo);
      if (!manifest) {
        cached[repo] = { bytes: 0, complete: false };
        continue;
      }
      const prefix = weightsKeyBase(repo, manifest.hash) + "/";
      const chunks = new Set<number>();
      let chunkSize = 0;
      for (const k of keys) {
        const m = k.url.startsWith(prefix) ? /chunk-(\d+)-(\d+)$/.exec(k.url) : null;
        if (!m) continue;
        chunkSize = Number(m[1]);
        chunks.add(Number(m[2]));
      }
      let bytes = 0;
      for (const c of chunks) bytes += Math.max(0, Math.min(chunkSize, manifest.size - c * chunkSize));
      const total = chunkSize ? Math.ceil(manifest.size / chunkSize) : Infinity;
      cached[repo] = { bytes, complete: chunks.size === total };
    }
  } catch {
    // Cache API unavailable
  }
  let est: StorageEstimate = {};
  try {
    est = (await navigator.storage?.estimate?.()) ?? {};
  } catch {
    // ignore
  }
  post({ type: "cacheStatus", cached, usage: est.usage ?? 0, quota: est.quota ?? 0 });
}

async function clearCache(repo?: string) {
  if (typeof caches === "undefined") return;
  if (!repo) {
    await caches.delete(CACHE_NAME);
    await caches.delete(FILES_CACHE);
    return;
  }
  const prefix = `${KEY_ORIGIN}/${repo}/`;
  for (const name of [CACHE_NAME, FILES_CACHE]) {
    const cache = await openCache(name);
    for (const k of (await cache?.keys()) ?? []) {
      if (k.url.startsWith(prefix)) await cache!.delete(k);
    }
  }
}

ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  if (msg.type === "stop") {
    stopRequested = true;
    return;
  }
  // Reset on arrival (not when the queued request starts), so that a stop sent
  // while this request waits in the queue still applies to it.
  if (msg.type === "generate") stopRequested = false;
  // Serialize everything else.
  busy = busy.then(async () => {
    try {
      switch (msg.type) {
        case "load":
          await load(msg.modelId, msg.repo, msg.fallbackBytes);
          break;
        case "generate":
          try {
            await generate(msg.id, msg.messages, msg.params);
          } catch (e) {
            post({ type: "genError", id: msg.id, message: errorMessage(e) });
          }
          break;
        case "unload":
          model?.destroy();
          model = null;
          loadedId = "";
          break;
        case "cacheStatus":
          await cacheStatus(msg.repos);
          break;
        case "clearCache":
          await clearCache(msg.repo);
          post({ type: "cleared" });
          break;
      }
    } catch (e) {
      if (msg.type === "load") {
        post({ type: "loadError", message: errorMessage(e) });
      } else {
        post({ type: "status", text: errorMessage(e) });
      }
    }
  });
};

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
