/**
 * WebGPU inference for the tiny-lm models: a Llama-style decoder (v1) and the
 * same decoder with Megatron-style MoE MLPs (v3). Weights are used directly in
 * bf16 from the safetensors file; activations and the KV cache are f32.
 */
import {
  attentionShader,
  embedShader,
  gemmShader,
  gemvShader,
  moeCombineShader,
  moeGroupShader,
  rmsnormShader,
  ropeCacheShader,
  topkShader,
} from "./shaders.ts";
import { type ChunkSource, type Placement, type TensorInfo, readSafetensorsHeader, uploadTensors } from "./weights.ts";

export interface ModelConfig {
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  vocab: number;
  ropeTheta: number;
  /** Llama 3 frequency scaling of RoPE, if the model uses it. */
  ropeLlama3: null | { factor: number; lowFreqFactor: number; highFreqFactor: number; originalMaxPosition: number };
  eps: number;
  maxSeq: number;
  /** Dense MLP width (dense models). */
  intermediate: number;
  moe: null | { experts: number; topk: number; intermediate: number; sharedIntermediate: number };
}

// biome-ignore lint: config.json is loosely typed
export function parseConfig(cfg: any): ModelConfig {
  const moe =
    cfg.num_experts && cfg.num_experts > 0
      ? {
          experts: cfg.num_experts,
          topk: cfg.num_experts_per_tok,
          intermediate: cfg.moe_intermediate_size ?? cfg.intermediate_size,
          sharedIntermediate: cfg.shared_expert_intermediate_size ?? 0,
        }
      : null;
  const rope = cfg.rope_parameters ?? cfg.rope_scaling ?? null;
  const ropeType = rope?.rope_type ?? rope?.type ?? "default";
  if (ropeType !== "default" && ropeType !== "llama3") throw new Error(`Unsupported rope_type ${ropeType}`);
  return {
    hidden: cfg.hidden_size,
    layers: cfg.num_hidden_layers,
    heads: cfg.num_attention_heads,
    kvHeads: cfg.num_key_value_heads ?? cfg.num_attention_heads,
    headDim: cfg.head_dim ?? cfg.hidden_size / cfg.num_attention_heads,
    vocab: cfg.vocab_size,
    ropeTheta: rope?.rope_theta ?? cfg.rope_theta ?? 10000,
    ropeLlama3:
      ropeType === "llama3"
        ? {
            factor: rope.factor,
            lowFreqFactor: rope.low_freq_factor,
            highFreqFactor: rope.high_freq_factor,
            originalMaxPosition: rope.original_max_position_embeddings,
          }
        : null,
    eps: cfg.rms_norm_eps ?? 1e-5,
    maxSeq: Math.min(cfg.max_position_embeddings ?? 2048, 2048),
    intermediate: cfg.intermediate_size,
    moe,
  };
}

const f32 = Math.fround;

/**
 * RoPE inverse frequencies, 1 / theta^(2i/d), with the optional Llama 3 scaling
 * (same as HF `_compute_llama3_parameters`), each step rounded to fp32.
 */
export function ropeInvFreq(cfg: ModelConfig): Float32Array {
  const half = cfg.headDim / 2;
  const out = new Float32Array(half);
  for (let j = 0; j < half; j++) {
    const exponent = f32(f32(2 * j) / cfg.headDim);
    out[j] = f32(1 / f32(cfg.ropeTheta ** exponent));
  }
  const s = cfg.ropeLlama3;
  if (!s) return out;
  const lowFreqWavelen = s.originalMaxPosition / s.lowFreqFactor;
  const highFreqWavelen = s.originalMaxPosition / s.highFreqFactor;
  for (let j = 0; j < half; j++) {
    const invFreq = out[j];
    const wavelen = f32(f32(2 * Math.PI) / invFreq);
    const scaled = wavelen > lowFreqWavelen ? f32(invFreq / s.factor) : invFreq;
    if (wavelen <= lowFreqWavelen && wavelen >= highFreqWavelen) {
      const smooth = f32(f32(f32(s.originalMaxPosition / wavelen) - s.lowFreqFactor) / (s.highFreqFactor - s.lowFreqFactor));
      out[j] = f32(f32(f32(f32(1 - smooth) * scaled) / s.factor) + f32(smooth * scaled));
    } else {
      out[j] = scaled;
    }
  }
  return out;
}

interface LayerWeights {
  ln1: GPUBuffer;
  ln2: GPUBuffer;
  qkv: GPUBuffer;
  o: GPUBuffer;
  // dense
  gate?: GPUBuffer;
  up?: GPUBuffer;
  down?: GPUBuffer;
  // moe
  router?: GPUBuffer;
  eGate?: GPUBuffer;
  eUp?: GPUBuffer;
  eDown?: GPUBuffer;
  sGate?: GPUBuffer;
  sUp?: GPUBuffer;
  sDown?: GPUBuffer;
  kc: GPUBuffer;
  vc: GPUBuffer;
}

interface Op {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  wg: [number, number, number];
}

interface Program {
  ops: Op[];
  uniform: GPUBuffer;
}

const ALIGN = 256;
const STORAGE = 0x80; // GPUBufferUsage.STORAGE
const COPY_SRC = 0x04;
const COPY_DST = 0x08;
const UNIFORM = 0x40;
const MAP_READ = 0x01;

export interface LoadProgress {
  doneBytes: number;
  totalBytes: number;
}

export class TinyLM {
  readonly cfg: ModelConfig;
  readonly device: GPUDevice;
  /** Largest number of tokens processed per forward call. */
  readonly chunk: number;
  private embedShards: { buffer: GPUBuffer; rowStart: number; rowEnd: number }[] = [];
  private finalNorm!: GPUBuffer;
  private layers: LayerWeights[] = [];
  private rope!: GPUBuffer;
  private state!: GPUBuffer;
  private ids!: GPUBuffer;
  private x!: GPUBuffer;
  private xn!: GPUBuffer;
  private qkv!: GPUBuffer;
  private attn!: GPUBuffer;
  private mlpH!: GPUBuffer;
  private moeH!: GPUBuffer;
  private moeY!: GPUBuffer;
  private routerLogits!: GPUBuffer;
  private sel!: GPUBuffer;
  private selW!: GPUBuffer;
  private groups!: GPUBuffer;
  private logits!: GPUBuffer;
  private readback!: GPUBuffer;
  private pipelines = new Map<string, GPUComputePipeline>();
  private programs = new Map<string, Program>();
  private allBuffers: GPUBuffer[] = [];
  /** Tokens whose keys/values are currently in the KV cache. */
  private cachedTokens: number[] = [];
  /** How many prompt tokens the last `prefill` call took from the KV cache. */
  lastReused = 0;

  private constructor(device: GPUDevice, cfg: ModelConfig, chunk: number) {
    this.device = device;
    this.cfg = cfg;
    this.chunk = chunk;
  }

  static async load(
    device: GPUDevice,
    configJson: unknown,
    src: ChunkSource,
    opts: { onProgress?: (p: LoadProgress) => void; signal?: AbortSignal; chunk?: number } = {},
  ): Promise<TinyLM> {
    const cfg = parseConfig(configJson);
    if (cfg.headDim !== 64 && cfg.headDim !== 128) throw new Error(`Unsupported head_dim ${cfg.headDim}`);
    if (cfg.moe && (cfg.moe.experts > 16 || cfg.moe.topk > 8)) throw new Error("Unsupported MoE shape");
    const m = new TinyLM(device, cfg, opts.chunk ?? 256);
    const tensors = await readSafetensorsHeader(src, opts.signal);
    let placement: Map<string, Placement[]>;
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    try {
      placement = m.allocate(tensors);
      m.allocateRuntime();
    } catch (e) {
      m.destroy();
      throw e;
    } finally {
      const validation = await device.popErrorScope();
      const oom = await device.popErrorScope();
      if (oom || validation) {
        m.destroy();
        // biome-ignore lint: rethrowing a clearer error from finally is intended
        throw new Error(
          oom
            ? "GPU メモリが足りません。他のタブやアプリを閉じるか、小さいモデルを試してください。"
            : `GPU バッファを確保できませんでした: ${validation!.message}`,
        );
      }
    }
    try {
      const unused = tensors.filter((t) => !placement.has(t.name)).map((t) => t.name);
      if (unused.length) throw new Error(`Unexpected tensors: ${unused.slice(0, 5).join(", ")}`);
      await uploadTensors(device, src, tensors, (t) => placement.get(t.name), {
        onProgress: (doneBytes, totalBytes) => opts.onProgress?.({ doneBytes, totalBytes }),
        signal: opts.signal,
      });
      await device.queue.onSubmittedWorkDone();
    } catch (e) {
      m.destroy();
      throw e;
    }
    return m;
  }

  private buf(size: number, usage = STORAGE | COPY_DST, label?: string): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label });
    this.allBuffers.push(b);
    return b;
  }

  /** Creates weight buffers and maps every tensor of the file to its destination. */
  private allocate(tensors: TensorInfo[]): Map<string, Placement[]> {
    const { cfg } = this;
    const byName = new Map(tensors.map((t) => [t.name, t]));
    const need = (name: string) => {
      const t = byName.get(name);
      if (!t) throw new Error(`Missing tensor ${name}`);
      if (t.dtype !== "BF16") throw new Error(`Expected BF16 for ${name}, got ${t.dtype}`);
      return t;
    };
    const place = new Map<string, Placement[]>();
    const bytes = (t: TensorInfo) => t.end - t.start;
    const single = (name: string, label: string) => {
      const t = need(name);
      const b = this.buf(bytes(t), STORAGE | COPY_DST, label);
      place.set(name, [{ buffer: b, offset: 0, from: 0, to: bytes(t) }]);
      return b;
    };
    const fused = (names: string[], label: string) => {
      const ts = names.map(need);
      const b = this.buf(
        ts.reduce((s, t) => s + bytes(t), 0),
        STORAGE | COPY_DST,
        label,
      );
      let off = 0;
      for (const t of ts) {
        place.set(t.name, [{ buffer: b, offset: off, from: 0, to: bytes(t) }]);
        off += bytes(t);
      }
      return b;
    };

    // Embedding (tied with the LM head), split into shards that fit the binding limit.
    const emb = need("model.embed_tokens.weight");
    const rowBytes = cfg.hidden * 2;
    const maxBinding = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    const rowsPerShard = Math.floor(maxBinding / rowBytes / 8) * 8;
    if (emb.shape[0] !== cfg.vocab) throw new Error("vocab_size does not match the embedding");
    const embPieces: Placement[] = [];
    for (let r = 0; r < cfg.vocab; r += rowsPerShard) {
      const rowEnd = Math.min(cfg.vocab, r + rowsPerShard);
      const buffer = this.buf((rowEnd - r) * rowBytes, STORAGE | COPY_DST, "embed");
      this.embedShards.push({ buffer, rowStart: r, rowEnd });
      embPieces.push({ buffer, offset: 0, from: r * rowBytes, to: rowEnd * rowBytes });
    }
    place.set(emb.name, embPieces);
    this.finalNorm = single("model.norm.weight", "norm");

    for (let i = 0; i < cfg.layers; i++) {
      const p = `model.layers.${i}.`;
      const kvSize = cfg.kvHeads * cfg.maxSeq * cfg.headDim * 4;
      const L: LayerWeights = {
        ln1: single(p + "input_layernorm.weight", `ln1.${i}`),
        ln2: single(p + "post_attention_layernorm.weight", `ln2.${i}`),
        qkv: fused([p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"], `qkv.${i}`),
        o: single(p + "self_attn.o_proj.weight", `o.${i}`),
        kc: this.buf(kvSize, STORAGE, `kc.${i}`),
        vc: this.buf(kvSize, STORAGE, `vc.${i}`),
      };
      if (cfg.moe) {
        const E = cfg.moe.experts;
        const ex = (proj: string) => Array.from({ length: E }, (_, e) => `${p}mlp.experts.${e}.${proj}.weight`);
        L.router = single(p + "mlp.gate.weight", `router.${i}`);
        L.eGate = fused(ex("gate_proj"), `eGate.${i}`);
        L.eUp = fused(ex("up_proj"), `eUp.${i}`);
        L.eDown = fused(ex("down_proj"), `eDown.${i}`);
        if (cfg.moe.sharedIntermediate > 0) {
          L.sGate = single(p + "mlp.shared_expert.gate_proj.weight", `sGate.${i}`);
          L.sUp = single(p + "mlp.shared_expert.up_proj.weight", `sUp.${i}`);
          L.sDown = single(p + "mlp.shared_expert.down_proj.weight", `sDown.${i}`);
        }
      } else {
        L.gate = single(p + "mlp.gate_proj.weight", `gate.${i}`);
        L.up = single(p + "mlp.up_proj.weight", `up.${i}`);
        L.down = single(p + "mlp.down_proj.weight", `down.${i}`);
      }
      this.layers.push(L);
    }
    return place;
  }

  private allocateRuntime() {
    const { cfg, chunk: C } = this;
    const H = cfg.hidden;
    const qkvW = (cfg.heads + 2 * cfg.kvHeads) * cfg.headDim;
    const R = cfg.moe ? C * cfg.moe.topk : 0;
    this.state = this.buf(16, STORAGE | COPY_DST, "state");
    this.ids = this.buf(C * 4, STORAGE | COPY_DST, "ids");
    this.x = this.buf(C * H * 4, STORAGE, "x");
    this.xn = this.buf(C * H * 4, STORAGE, "xn");
    this.qkv = this.buf(C * qkvW * 4, STORAGE, "qkv");
    this.attn = this.buf(C * cfg.heads * cfg.headDim * 4, STORAGE, "attn");
    const mlpW = cfg.moe ? Math.max(cfg.moe.sharedIntermediate, 1) : cfg.intermediate;
    this.mlpH = this.buf(C * mlpW * 4, STORAGE, "mlpH");
    if (cfg.moe) {
      this.moeH = this.buf(R * cfg.moe.intermediate * 4, STORAGE, "moeH");
      this.moeY = this.buf(R * H * 4, STORAGE, "moeY");
      this.routerLogits = this.buf(C * cfg.moe.experts * 4, STORAGE, "router");
      this.sel = this.buf(R * 4, STORAGE, "sel");
      this.selW = this.buf(R * 4, STORAGE, "selW");
      this.groups = this.buf((32 + R) * 4, STORAGE, "groups");
    }
    this.logits = this.buf(cfg.vocab * 4, STORAGE | COPY_SRC, "logits");
    this.readback = this.buf(cfg.vocab * 4, MAP_READ | COPY_DST, "readback");

    // RoPE table: angle = pos * inv_freq, computed in fp32 like PyTorch.
    const half = cfg.headDim / 2;
    const invFreqs = ropeInvFreq(cfg);
    const table = new Float32Array(2 * cfg.maxSeq * half);
    for (let j = 0; j < half; j++) {
      const invFreq = invFreqs[j];
      for (let pos = 0; pos < cfg.maxSeq; pos++) {
        const a = Math.fround(pos * invFreq);
        table[pos * half + j] = Math.cos(a);
        table[cfg.maxSeq * half + pos * half + j] = Math.sin(a);
      }
    }
    this.rope = this.buf(table.byteLength, STORAGE | COPY_DST, "rope");
    this.device.queue.writeBuffer(this.rope, 0, table);
  }

  private pipeline(key: string, code: string): GPUComputePipeline {
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code, label: key }), entryPoint: "main" },
        label: key,
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** Builds the list of dispatches for one forward pass over T tokens. */
  private buildProgram(T: number): Program {
    const { cfg, device } = this;
    const H = cfg.hidden;
    const D = cfg.headDim;
    const qkvW = (cfg.heads + 2 * cfg.kvHeads) * D;
    const attnW = cfg.heads * D;
    const params: ArrayBuffer[] = [];
    const pending: { pipeline: GPUComputePipeline; entries: [number, GPUBufferBinding][]; wg: [number, number, number]; slot: number }[] = [];

    const u32s = (...v: number[]) => new Uint32Array(v).buffer;
    const add = (
      pipeline: GPUComputePipeline,
      uniform: ArrayBuffer,
      entries: [number, GPUBuffer | GPUBufferBinding][],
      wg: [number, number, number],
    ) => {
      params.push(uniform);
      pending.push({
        pipeline,
        entries: entries.map(([b, r]) => [b, "buffer" in r ? r : { buffer: r }]),
        wg,
        slot: params.length - 1,
      });
    };

    const mm = (
      x: GPUBuffer | GPUBufferBinding,
      W: GPUBuffer,
      y: GPUBuffer,
      M: number,
      N: number,
      K: number,
      o: {
        accumulate?: boolean;
        W2?: GPUBuffer;
        xStride?: number;
        yStride?: number;
        nOffset?: number;
        routed?: { table: GPUBuffer; topk: number; expertStrideElems: number; groupRows?: number };
      } = {},
    ) => {
      const accumulate = !!o.accumulate;
      const swiglu = !!o.W2;
      const xStride = o.xStride ?? K;
      const yStride = o.yStride ?? N;
      const nOffset = o.nOffset ?? 0;
      const routed = o.routed;
      const small = routed ? routed.groupRows === undefined : M <= 4;
      if (small) {
        const key = `gemv:${accumulate}:${!!routed}:${swiglu}`;
        const pipe = this.pipeline(key, gemvShader({ accumulate, routed: !!routed, swiglu }));
        const entries: [number, GPUBuffer | GPUBufferBinding][] = [
          [1, x],
          [2, W],
          [3, y],
        ];
        if (swiglu) entries.push([4, o.W2!]);
        if (routed) entries.push([5, routed.table]);
        add(
          pipe,
          u32s(M, N, K, xStride, yStride, nOffset, routed?.topk ?? 1, routed ? routed.expertStrideElems / 8 : 0),
          entries,
          [Math.ceil(N / 8), M, 1],
        );
      } else {
        const grouped = !!routed;
        const key = `gemm:${accumulate}:${grouped}:${swiglu}`;
        const pipe = this.pipeline(key, gemmShader({ accumulate, grouped, swiglu }));
        const entries: [number, GPUBuffer | GPUBufferBinding][] = [
          [1, x],
          [2, W],
          [3, y],
        ];
        if (swiglu) entries.push([4, o.W2!]);
        if (routed) entries.push([5, routed.table]);
        const rowsY = grouped ? routed!.groupRows! : M;
        add(
          pipe,
          u32s(M, N, K, xStride, yStride, nOffset, routed?.topk ?? 1, routed ? routed.expertStrideElems / 2 : 0),
          entries,
          [Math.ceil(N / 64), Math.ceil(rowsY / 64), grouped ? cfg.moe!.experts : 1],
        );
      }
    };

    const rmsnorm = (x: GPUBuffer, w: GPUBuffer, out: GPUBuffer, rows: number, inRow = 0) => {
      const pipe = this.pipeline("rmsnorm", rmsnormShader);
      const u = new ArrayBuffer(16);
      new Uint32Array(u, 0, 1)[0] = H;
      new Float32Array(u, 4, 1)[0] = cfg.eps;
      new Uint32Array(u, 8, 2).set([inRow * H, 0]);
      add(pipe, u, [[1, x], [2, w], [3, out]], [rows, 1, 1]);
    };

    // Embedding
    for (const s of this.embedShards) {
      add(
        this.pipeline("embed", embedShader),
        u32s(T, H, s.rowStart, s.rowEnd),
        [[1, this.ids], [2, s.buffer], [3, this.x]],
        [Math.ceil((T * H) / 2 / 256), 1, 1],
      );
    }

    for (const L of this.layers) {
      // Attention block
      rmsnorm(this.x, L.ln1, this.xn, T);
      mm(this.xn, L.qkv, this.qkv, T, qkvW, H);
      add(
        this.pipeline("rope", ropeCacheShader),
        u32s(T, cfg.heads, cfg.kvHeads, D, cfg.maxSeq, qkvW, 0, 0),
        [[1, this.qkv], [2, this.rope], [3, L.kc], [4, L.vc], [5, this.state]],
        [Math.ceil((T * (cfg.heads + 2 * cfg.kvHeads) * (D / 2)) / 256), 1, 1],
      );
      {
        const u = new ArrayBuffer(32);
        new Uint32Array(u, 0, 7).set([T, cfg.heads, cfg.kvHeads, D, cfg.maxSeq, qkvW, attnW]);
        new Float32Array(u, 28, 1)[0] = 1 / Math.sqrt(D);
        add(
          this.pipeline(`attn:${D}`, attentionShader(D)),
          u,
          [[1, this.qkv], [2, L.kc], [3, L.vc], [4, this.attn], [5, this.state]],
          [cfg.heads, T, 1],
        );
      }
      mm(this.attn, L.o, this.x, T, H, attnW, { accumulate: true });

      // MLP block
      rmsnorm(this.x, L.ln2, this.xn, T);
      if (!cfg.moe) {
        mm(this.xn, L.gate!, this.mlpH, T, cfg.intermediate, H, { W2: L.up! });
        mm(this.mlpH, L.down!, this.x, T, H, cfg.intermediate, { accumulate: true });
      } else {
        const { experts: E, topk, intermediate: I, sharedIntermediate: S } = cfg.moe;
        const R = T * topk;
        mm(this.xn, L.router!, this.routerLogits, T, E, H);
        add(
          this.pipeline("topk", topkShader),
          u32s(T, E, topk, 0),
          [[1, this.routerLogits], [2, this.sel], [3, this.selW]],
          [Math.ceil(T / 64), 1, 1],
        );
        const decode = T <= 2;
        if (decode) {
          mm(this.xn, L.eGate!, this.moeH, R, I, H, {
            W2: L.eUp!,
            routed: { table: this.sel, topk, expertStrideElems: I * H },
          });
          mm(this.moeH, L.eDown!, this.moeY, R, H, I, {
            routed: { table: this.sel, topk: 1, expertStrideElems: H * I },
          });
        } else {
          add(this.pipeline("moeGroup", moeGroupShader), u32s(R, E, 0, 0), [[1, this.sel], [2, this.groups]], [1, 1, 1]);
          mm(this.xn, L.eGate!, this.moeH, R, I, H, {
            W2: L.eUp!,
            routed: { table: this.groups, topk, expertStrideElems: I * H, groupRows: T },
          });
          mm(this.moeH, L.eDown!, this.moeY, R, H, I, {
            routed: { table: this.groups, topk: 1, expertStrideElems: H * I, groupRows: T },
          });
        }
        add(
          this.pipeline("moeCombine", moeCombineShader),
          u32s(T, H, topk, 0),
          [[1, this.moeY], [2, this.selW], [3, this.x]],
          [Math.ceil((T * H) / 256), 1, 1],
        );
        if (S > 0) {
          mm(this.xn, L.sGate!, this.mlpH, T, S, H, { W2: L.sUp! });
          mm(this.mlpH, L.sDown!, this.x, T, H, S, { accumulate: true });
        }
      }
    }

    // Final norm and LM head on the last token only.
    rmsnorm(this.x, this.finalNorm, this.xn, 1, T - 1);
    for (const s of this.embedShards) {
      mm(this.xn, s.buffer, this.logits, 1, s.rowEnd - s.rowStart, H, { nOffset: s.rowStart, yStride: cfg.vocab });
    }

    // Pack all uniforms into one buffer.
    const uniform = device.createBuffer({ size: Math.max(ALIGN, params.length * ALIGN), usage: UNIFORM | COPY_DST });
    const arena = new Uint8Array(params.length * ALIGN);
    params.forEach((p, i) => arena.set(new Uint8Array(p), i * ALIGN));
    device.queue.writeBuffer(uniform, 0, arena);
    const ops: Op[] = pending.map((op) => ({
      pipeline: op.pipeline,
      wg: op.wg,
      bindGroup: device.createBindGroup({
        layout: op.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform, offset: op.slot * ALIGN, size: params[op.slot].byteLength } },
          ...op.entries.map(([binding, resource]) => ({ binding, resource })),
        ],
      }),
    }));
    return { ops, uniform };
  }

  private program(T: number): Program {
    const key = String(T);
    let p = this.programs.get(key);
    if (!p) {
      p = this.buildProgram(T);
      this.programs.set(key, p);
      if (this.programs.size > 8) {
        // Evict the oldest program other than the decode one (T = 1).
        for (const k of this.programs.keys()) {
          if (k === "1" || k === key) continue;
          this.programs.get(k)!.uniform.destroy();
          this.programs.delete(k);
          break;
        }
      }
    }
    return p;
  }

  /** Runs tokens at positions [pos0, pos0 + T) and returns the logits of the last one. */
  private async forward(tokens: number[], pos0: number): Promise<Float32Array> {
    const T = tokens.length;
    if (T < 1 || T > this.chunk) throw new Error("bad chunk length");
    if (pos0 + T > this.cfg.maxSeq) throw new Error("context length exceeded");
    const prog = this.program(T);
    const q = this.device.queue;
    q.writeBuffer(this.ids, 0, new Uint32Array(tokens));
    q.writeBuffer(this.state, 0, new Uint32Array([pos0, 0, 0, 0]));
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const op of prog.ops) {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      pass.dispatchWorkgroups(op.wg[0], op.wg[1], op.wg[2]);
    }
    pass.end();
    enc.copyBufferToBuffer(this.logits, 0, this.readback, 0, this.cfg.vocab * 4);
    q.submit([enc.finish()]);
    await this.readback.mapAsync(1 /* GPUMapMode.READ */);
    const out = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return out;
  }

  /**
   * Feeds `tokens` (the full sequence so far), reusing the KV cache for the
   * longest common prefix with what was processed before. Returns the logits
   * for the next token.
   */
  async prefill(tokens: number[], onProgress?: (done: number, total: number) => void): Promise<Float32Array> {
    if (tokens.length === 0) throw new Error("empty prompt");
    if (tokens.length > this.cfg.maxSeq) throw new Error("prompt too long");
    let common = 0;
    const prev = this.cachedTokens;
    while (common < prev.length && common < tokens.length && prev[common] === tokens[common]) common++;
    // At least one token must be run to get logits.
    if (common === tokens.length) common--;
    this.lastReused = common;
    let logits: Float32Array | null = null;
    this.cachedTokens = tokens.slice(0, common);
    for (let p = common; p < tokens.length; p += this.chunk) {
      const part = tokens.slice(p, Math.min(tokens.length, p + this.chunk));
      logits = await this.forward(part, p);
      this.cachedTokens.push(...part);
      onProgress?.(p + part.length - common, tokens.length - common);
    }
    return logits!;
  }

  /** Runs one generated token and returns the logits for the next. */
  async step(token: number): Promise<Float32Array> {
    const pos = this.cachedTokens.length;
    const logits = await this.forward([token], pos);
    this.cachedTokens.push(token);
    return logits;
  }

  get position(): number {
    return this.cachedTokens.length;
  }

  resetCache(): void {
    this.cachedTokens = [];
  }

  destroy(): void {
    for (const b of this.allBuffers) b.destroy();
    for (const p of this.programs.values()) p.uniform.destroy();
    this.allBuffers = [];
    this.programs.clear();
  }
}
