// End-to-end check of the WebGPU engine against PyTorch reference outputs.
// Usage: node tests/model.test.ts <model dir with config.json, model.safetensors, tokenizer.json> <ref prefix>
import fs from "node:fs";
import path from "node:path";
import { create, globals } from "webgpu";
import { requestDevice } from "../src/engine/device.ts";
import { TinyLM } from "../src/engine/model.ts";
import { UnigramTokenizer, renderChatTemplate } from "../src/engine/tokenizer.ts";
import type { ChunkSource } from "../src/engine/weights.ts";

Object.assign(globalThis, globals);
// Keep the dawn instance reachable: if it is garbage collected, pending GPU events crash the process.
const gpu = create([]) as unknown as GPU;
(globalThis as { __dawnInstance?: unknown }).__dawnInstance = gpu;

class FileChunkSource implements ChunkSource {
  readonly size: number;
  readonly chunkSize = 32 * 1024 * 1024;
  private fd: number;
  constructor(file: string) {
    this.fd = fs.openSync(file, "r");
    this.size = fs.fstatSync(this.fd).size;
  }
  async readChunk(i: number): Promise<Uint8Array> {
    const start = i * this.chunkSize;
    const len = Math.min(this.chunkSize, this.size - start);
    const buf = new Uint8Array(len);
    fs.readSync(this.fd, buf, 0, len, start);
    return buf;
  }
}

const [dir, refPrefix] = process.argv.slice(2);
const ref = JSON.parse(fs.readFileSync(refPrefix + ".json", "utf8"));
const { device, adapterInfo } = await requestDevice(gpu);
device.addEventListener?.("uncapturederror", (e) => console.error("GPU error:", (e as GPUUncapturedErrorEvent).error.message));
console.log("adapter:", adapterInfo, "maxBinding", device.limits.maxStorageBufferBindingSize);
const tok = new UnigramTokenizer(JSON.parse(fs.readFileSync(path.join(dir, "tokenizer.json"), "utf8")));
const cfg = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8"));
let t0 = performance.now();
const model = await TinyLM.load(device, cfg, new FileChunkSource(path.join(dir, "model.safetensors")));
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

let failures = 0;
for (let c = 0; c < ref.length; c++) {
  const r = ref[c];
  const ids = tok.encode(renderChatTemplate(r.messages, true));
  if (JSON.stringify(ids) !== JSON.stringify(r.input_ids)) {
    console.log("input ids differ", ids, r.input_ids);
    failures++;
  }
  model.resetCache();
  t0 = performance.now();
  const logits = await model.prefill(r.input_ids);
  const tPrefill = performance.now() - t0;
  const refLast = new Float32Array(fs.readFileSync(`${refPrefix}_case${c}_last.f32`).buffer.slice(0));
  let maxDiff = 0, maxRef = 0;
  for (let i = 0; i < refLast.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(logits[i] - refLast[i]));
    maxRef = Math.max(maxRef, Math.abs(refLast[i]));
  }
  const argmax = (a: Float32Array) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
  console.log(`case ${c}: prefill ${r.input_ids.length} tok in ${tPrefill.toFixed(0)}ms, max|diff| ${maxDiff.toExponential(2)} (max|ref| ${maxRef.toFixed(2)}), argmax ${argmax(logits)} vs ${argmax(refLast)}`);
  if (maxDiff > 1e-2 * Math.max(1, maxRef)) failures++;

  // Greedy decode
  const gen: number[] = [];
  let cur = logits;
  t0 = performance.now();
  for (let i = 0; i < r.greedy.length; i++) {
    const next = argmax(cur);
    gen.push(next);
    if (next === tok.eosId) break;
    if (process.env.DEBUG) console.log("step", i, next);
    cur = await model.step(next);
  }
  const dt = performance.now() - t0;
  const match = gen.every((g, i) => g === r.greedy[i]);
  let same = 0;
  while (same < gen.length && gen[same] === r.greedy[same]) same++;
  console.log(`  greedy ${match ? "MATCH" : `DIFFERS at ${same}`} (${gen.length} tok, ${(gen.length / (dt / 1000)).toFixed(1)} tok/s)`);
  console.log("  ts:", JSON.stringify(tok.decode(gen)));
  if (!match) { console.log("  py:", JSON.stringify(r.greedy_text)); if (same < 10) failures++; }

  // KV cache reuse: process a prefix first, then the full prompt (only the suffix is run).
  model.resetCache();
  await model.prefill(r.input_ids.slice(0, Math.ceil(r.input_ids.length / 2)));
  const reused = await model.prefill(r.input_ids);
  let reuseDiff = 0;
  for (let i = 0; i < refLast.length; i++) reuseDiff = Math.max(reuseDiff, Math.abs(reused[i] - refLast[i]));
  console.log(`  prefix reuse: max|diff| ${reuseDiff.toExponential(2)}`);
  if (reuseDiff > 1e-2 * Math.max(1, maxRef)) failures++;
}
model.destroy();
console.log(failures ? `FAILED (${failures})` : "OK");
process.exit(failures ? 1 : 0);
