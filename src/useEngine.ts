import { useCallback, useEffect, useRef, useState } from "react";
import type { GenerationStats, WorkerRequest, WorkerResponse } from "./engine/protocol.ts";
import type { SamplingParams } from "./engine/sampler.ts";
import type { ChatMessage } from "./engine/tokenizer.ts";
import { MODELS, type ModelEntry } from "./models.ts";

export type Phase = "idle" | "loading" | "ready" | "error";

export interface LoadState {
  phase: Phase;
  modelId: string | null;
  status: string;
  doneBytes: number;
  totalBytes: number;
  fromCache: boolean;
  startedAt: number;
  error: string | null;
  adapter: string;
}

export interface GenerateCallbacks {
  onToken: (text: string) => void;
  onPrefill?: (done: number, total: number) => void;
  onDone: (text: string, stats: GenerationStats) => void;
  onError: (message: string) => void;
}

const initialLoad: LoadState = {
  phase: "idle",
  modelId: null,
  status: "",
  doneBytes: 0,
  totalBytes: 0,
  fromCache: false,
  startedAt: 0,
  error: null,
  adapter: "",
};

/** Owns the inference worker and exposes load / generate / stop. */
export function useEngine() {
  const workerRef = useRef<Worker | null>(null);
  const [load, setLoad] = useState<LoadState>(initialLoad);
  const [cached, setCached] = useState<Record<string, { bytes: number; complete: boolean }>>({});
  const [storage, setStorage] = useState({ usage: 0, quota: 0 });
  const genRef = useRef<{ id: number; cb: GenerateCallbacks } | null>(null);
  const nextId = useRef(1);

  const send = useCallback((m: WorkerRequest) => workerRef.current?.postMessage(m), []);

  useEffect(() => {
    const w = new Worker(new URL("./engine/worker.ts", import.meta.url), { type: "module" });
    workerRef.current = w;
    w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const m = ev.data;
      const g = genRef.current;
      switch (m.type) {
        case "status":
          setLoad((s) => ({ ...s, status: m.text }));
          break;
        case "progress":
          setLoad((s) => ({ ...s, doneBytes: m.doneBytes, totalBytes: m.totalBytes, fromCache: m.fromCache }));
          break;
        case "ready":
          setLoad((s) => ({ ...s, phase: "ready", modelId: m.modelId, status: "", adapter: m.adapter, error: null }));
          w.postMessage({ type: "cacheStatus", repos: MODELS.map((x) => x.repo) } satisfies WorkerRequest);
          break;
        case "loadError":
          setLoad((s) => ({ ...s, phase: "error", error: m.message, status: "" }));
          break;
        case "token":
          if (g && g.id === m.id) g.cb.onToken(m.text);
          break;
        case "prefill":
          if (g && g.id === m.id) g.cb.onPrefill?.(m.done, m.total);
          break;
        case "done":
          if (g && g.id === m.id) {
            genRef.current = null;
            g.cb.onDone(m.text, m.stats);
          }
          break;
        case "genError":
          if (g && g.id === m.id) {
            genRef.current = null;
            g.cb.onError(m.message);
          }
          break;
        case "cacheStatus":
          setCached(m.cached);
          setStorage({ usage: m.usage, quota: m.quota });
          break;
        case "cleared":
          w.postMessage({ type: "cacheStatus", repos: MODELS.map((x) => x.repo) } satisfies WorkerRequest);
          break;
      }
    };
    w.onerror = (e) => {
      setLoad((s) => ({ ...s, phase: "error", error: e.message || "ワーカーでエラーが発生しました" }));
    };
    w.postMessage({ type: "cacheStatus", repos: MODELS.map((x) => x.repo) } satisfies WorkerRequest);
    return () => {
      w.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadModel = useCallback(
    (entry: ModelEntry) => {
      // StorageManager.persist() is only available on the window, not in workers.
      navigator.storage?.persist?.().catch(() => {});
      setLoad({
        ...initialLoad,
        phase: "loading",
        modelId: entry.id,
        status: "準備しています",
        totalBytes: entry.weightBytes,
        startedAt: performance.now(),
      });
      send({ type: "load", modelId: entry.id, repo: entry.repo, fallbackBytes: entry.weightBytes });
    },
    [send],
  );

  const generate = useCallback(
    (messages: ChatMessage[], params: SamplingParams, cb: GenerateCallbacks) => {
      const id = nextId.current++;
      genRef.current = { id, cb };
      send({ type: "generate", id, messages, params });
    },
    [send],
  );

  const stop = useCallback(() => send({ type: "stop" }), [send]);
  const clearCache = useCallback((repo?: string) => send({ type: "clearCache", repo }), [send]);

  return { load, cached, storage, loadModel, generate, stop, clearCache };
}
