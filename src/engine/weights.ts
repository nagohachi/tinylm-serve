/**
 * Reading safetensors files in fixed-size chunks, so that a multi-GB file can be
 * streamed to the GPU without holding it in memory, and cached chunk by chunk.
 */

export interface ChunkSource {
  /** Total file size in bytes. */
  readonly size: number;
  readonly chunkSize: number;
  /** Returns the bytes of chunk `i` (the last chunk may be shorter). */
  readChunk(i: number, signal?: AbortSignal): Promise<Uint8Array>;
}

export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  /** Absolute byte range within the file. */
  start: number;
  end: number;
}

export async function readSafetensorsHeader(src: ChunkSource, signal?: AbortSignal): Promise<TensorInfo[]> {
  const first = await src.readChunk(0, signal);
  const view = new DataView(first.buffer, first.byteOffset, first.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  const dataStart = 8 + headerLen;
  let headerBytes: Uint8Array;
  if (dataStart <= first.byteLength) {
    headerBytes = first.subarray(8, dataStart);
  } else {
    headerBytes = new Uint8Array(headerLen);
    let filled = 0;
    for (let i = 0; filled < headerLen; i++) {
      const c = i === 0 ? first : await src.readChunk(i, signal);
      const from = i === 0 ? 8 : 0;
      const n = Math.min(c.byteLength - from, headerLen - filled);
      headerBytes.set(c.subarray(from, from + n), filled);
      filled += n;
    }
  }
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const tensors: TensorInfo[] = [];
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    tensors.push({
      name,
      dtype: t.dtype,
      shape: t.shape,
      start: dataStart + t.data_offsets[0],
      end: dataStart + t.data_offsets[1],
    });
  }
  tensors.sort((a, b) => a.start - b.start);
  return tensors;
}

/** Destination of the tensor bytes [from, to) (relative to the tensor start). */
export interface Placement {
  buffer: GPUBuffer;
  offset: number;
  from: number;
  to: number;
}

/**
 * Streams every chunk of the file and writes the tensor bytes into their GPU
 * placements. Tensors spanning chunk boundaries are written piecewise.
 */
export async function uploadTensors(
  device: GPUDevice,
  src: ChunkSource,
  tensors: TensorInfo[],
  place: (t: TensorInfo) => Placement[] | undefined,
  opts: { concurrency?: number; onProgress?: (doneBytes: number, totalBytes: number) => void; signal?: AbortSignal } = {},
): Promise<void> {
  // Flatten to absolute file ranges.
  const placed: { name: string; start: number; end: number; buffer: GPUBuffer; offset: number }[] = [];
  for (const t of tensors) {
    for (const p of place(t) ?? []) {
      placed.push({ name: t.name, start: t.start + p.from, end: t.start + p.to, buffer: p.buffer, offset: p.offset });
    }
  }
  if (!placed.length) return;
  placed.sort((a, b) => a.start - b.start);
  const firstByte = placed[0].start;
  const lastByte = Math.max(...placed.map((p) => p.end));
  const firstChunk = Math.floor(firstByte / src.chunkSize);
  const lastChunk = Math.floor((lastByte - 1) / src.chunkSize);
  const total = lastChunk - firstChunk + 1;
  const concurrency = opts.concurrency ?? 4;
  let next = firstChunk;
  let doneBytes = 0;
  const totalBytes = Math.min(src.size, (lastChunk + 1) * src.chunkSize) - firstChunk * src.chunkSize;

  // Abort the other in-flight downloads as soon as one chunk fails (or the caller aborts).
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort(opts.signal?.reason);
  opts.signal?.addEventListener("abort", onAbort);
  const worker = async () => {
    while (next <= lastChunk) {
      ctrl.signal.throwIfAborted();
      const i = next++;
      const data = await src.readChunk(i, ctrl.signal);
      const cStart = i * src.chunkSize;
      const cEnd = cStart + data.byteLength;
      for (const p of placed) {
        if (p.end <= cStart || p.start >= cEnd) continue;
        const from = Math.max(p.start, cStart);
        const to = Math.min(p.end, cEnd);
        const dst = p.offset + (from - p.start);
        if (dst % 4 !== 0 || (to - from) % 4 !== 0) {
          throw new Error(`Unaligned tensor piece for ${p.name}`);
        }
        device.queue.writeBuffer(p.buffer, dst, data.buffer, data.byteOffset + (from - cStart), to - from);
      }
      doneBytes += data.byteLength;
      opts.onProgress?.(doneBytes, totalBytes);
      // Keep the amount of staged upload data bounded.
      await device.queue.onSubmittedWorkDone();
    }
  };
  try {
    await Promise.all(
      Array.from({ length: Math.min(concurrency, total) }, () =>
        worker().catch((e) => {
          ctrl.abort(e);
          throw e;
        }),
      ),
    );
  } finally {
    opts.signal?.removeEventListener("abort", onAbort);
  }
}

/**
 * HTTP source using Range requests, with each chunk cached in the Cache API so
 * that later visits load from disk.
 */
export class HttpChunkSource implements ChunkSource {
  readonly url: string;
  readonly size: number;
  readonly chunkSize: number;
  private readonly cacheKeyBase: string;
  private readonly cache: Cache | null;
  private readonly onBytes?: (n: number) => void;
  private constructor(
    url: string,
    size: number,
    cacheKeyBase: string,
    cache: Cache | null,
    chunkSize: number,
    onBytes?: (n: number) => void,
  ) {
    this.url = url;
    this.size = size;
    this.cacheKeyBase = cacheKeyBase;
    this.cache = cache;
    this.chunkSize = chunkSize;
    this.onBytes = onBytes;
  }

  static async open(
    url: string,
    size: number,
    cacheName: string,
    cacheKeyBase: string,
    chunkSize = 32 * 1024 * 1024,
    onBytes?: (n: number) => void,
  ): Promise<HttpChunkSource> {
    let cache: Cache | null = null;
    try {
      if (typeof caches !== "undefined") cache = await caches.open(cacheName);
    } catch {
      cache = null;
    }
    return new HttpChunkSource(url, size, cacheKeyBase, cache, chunkSize, onBytes);
  }

  private key(i: number): string {
    return `${this.cacheKeyBase}/chunk-${this.chunkSize}-${i}`;
  }

  get chunkCount(): number {
    return Math.ceil(this.size / this.chunkSize);
  }

  async readChunk(i: number, signal?: AbortSignal): Promise<Uint8Array> {
    const start = i * this.chunkSize;
    const end = Math.min(this.size, start + this.chunkSize);
    const expected = end - start;
    if (this.cache) {
      try {
        const hit = await this.cache.match(this.key(i));
        if (hit) {
          const buf = new Uint8Array(await hit.arrayBuffer());
          if (buf.byteLength === expected) return buf;
        }
      } catch {
        // fall through to the network
      }
    }
    let lastErr: unknown;
    for (let attempt = 0; attempt < 6; attempt++) {
      try {
        signal?.throwIfAborted();
        const buf = await fetchRange(this.url, start, end, this.onBytes, signal);
        if (buf.byteLength !== expected) throw new Error(`Short read: ${buf.byteLength} / ${expected}`);
        if (this.cache) {
          try {
            await this.cache.put(
              this.key(i),
              new Response(buf as Uint8Array<ArrayBuffer>, { headers: { "Content-Type": "application/octet-stream" } }),
            );
          } catch {
            // Quota exceeded or caching unavailable: keep going without caching.
          }
        }
        return buf;
      } catch (e) {
        if (signal?.aborted) throw signal.reason ?? e;
        lastErr = e;
        console.warn(`chunk ${i}: attempt ${attempt + 1} failed:`, e);
        await new Promise((r) => setTimeout(r, 1000 * 2 ** Math.min(attempt, 4)));
      }
    }
    throw lastErr;
  }

  /** Number of chunks of this file that are already in the cache. */
  async cachedChunks(): Promise<number> {
    if (!this.cache) return 0;
    let c = 0;
    try {
      for (let i = 0; i < this.chunkCount; i++) if (await this.cache.match(this.key(i))) c++;
    } catch {
      // treat as not cached
    }
    return c;
  }
}

/** Abort a request that receives no data for this long, then retry it. */
const STALL_MS = 20_000;

/**
 * Fetches bytes [start, end) with a Range request. A watchdog aborts the
 * request when no data arrives for STALL_MS, so a stuck connection cannot
 * hang the whole download.
 */
async function fetchRange(
  url: string,
  start: number,
  end: number,
  onBytes?: (n: number) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort(signal?.reason);
  if (signal?.aborted) onAbort();
  signal?.addEventListener("abort", onAbort);
  let timer = setTimeout(() => ctrl.abort(new Error("download stalled")), STALL_MS);
  const kick = () => {
    clearTimeout(timer);
    timer = setTimeout(() => ctrl.abort(new Error("download stalled")), STALL_MS);
  };
  try {
    const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, signal: ctrl.signal });
    if (res.status !== 206) throw new Error(`HTTP ${res.status} while downloading ${url}`);
    if (!res.body) throw new Error("empty response body");
    const out = new Uint8Array(end - start);
    const reader = res.body.getReader();
    let filled = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      kick();
      if (filled + value.byteLength > out.byteLength) throw new Error("response longer than requested range");
      out.set(value, filled);
      filled += value.byteLength;
      onBytes?.(value.byteLength);
    }
    return filled === out.byteLength ? out : out.subarray(0, filled);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}
