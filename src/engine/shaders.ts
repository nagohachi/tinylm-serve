/**
 * WGSL kernels. Activations are f32; weights stay in bf16 exactly as stored in
 * the safetensors file, packed two per u32 (little endian: element 2i in the
 * low half, element 2i+1 in the high half).
 */

const BF16 = /* wgsl */ `
fn bf_lo(w: u32) -> f32 { return bitcast<f32>(w << 16u); }
fn bf_hi(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }
fn dot8(w: vec4<u32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
  return bf_lo(w.x) * a.x + bf_hi(w.x) * a.y + bf_lo(w.y) * a.z + bf_hi(w.y) * a.w
       + bf_lo(w.z) * b.x + bf_hi(w.z) * b.y + bf_lo(w.w) * b.z + bf_hi(w.w) * b.w;
}
`;

/** Embedding lookup for one shard of the vocabulary. */
export const embedShader = /* wgsl */ `
${BF16}
struct P { T: u32, H: u32, rowStart: u32, rowEnd: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let half = p.H / 2u;
  let i = g.x;
  if (i >= p.T * half) { return; }
  let t = i / half;
  let j = i % half;
  let id = ids[t];
  if (id < p.rowStart || id >= p.rowEnd) { return; }
  let w = table[(id - p.rowStart) * half + j];
  out[t * p.H + 2u * j] = bf_lo(w);
  out[t * p.H + 2u * j + 1u] = bf_hi(w);
}
`;

/** RMSNorm over rows of length H. One workgroup per row. */
export const rmsnormShader = /* wgsl */ `
${BF16}
struct P { H: u32, eps: f32, inOffset: u32, outOffset: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = wg.x;
  let base = p.inOffset + row * p.H;
  var s = 0.0;
  for (var i = l.x; i < p.H; i += 256u) { let v = x[base + i]; s += v * v; }
  red[l.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l.x < k) { red[l.x] += red[l.x + k]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.H) + p.eps);
  let obase = p.outOffset + row * p.H;
  for (var i = l.x; i < p.H; i += 256u) {
    let pw = w[i / 2u];
    let wv = select(bf_lo(pw), bf_hi(pw), (i & 1u) == 1u);
    out[obase + i] = (x[base + i] * r) * wv;
  }
}
`;

/**
 * y[m, n] (+)= sum_k x[m, k] * W[n, k] for small M (decode). Each workgroup
 * computes 8 output rows of W with 32 threads per row.
 *
 * mode 0: plain linear. mode 1: rows come from a (token, slot) routing table;
 * the expert id picks the weight block and the token picks the input row.
 */
export function gemvShader(opts: { accumulate: boolean; routed: boolean; swiglu: boolean }): string {
  const { accumulate, routed, swiglu } = opts;
  return /* wgsl */ `
${BF16}
struct P { M: u32, N: u32, K: u32, xStride: u32, yStride: u32, nOffset: u32, topk: u32, expertStride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${swiglu ? "@group(0) @binding(4) var<storage, read> W2: array<vec4<u32>>;" : ""}
${routed ? "@group(0) @binding(5) var<storage, read> sel: array<u32>;" : ""}
var<workgroup> red: array<f32, 256>;
${swiglu ? "var<workgroup> red2: array<f32, 256>;" : ""}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lane = l.x % 32u;
  let r = l.x / 32u;
  let n = wg.x * 8u + r;
  let m = wg.y;
  let K8 = p.K / 8u;
  ${routed ? "let xrow = m / p.topk; let wbase = sel[m] * p.expertStride;" : "let xrow = m; let wbase = 0u;"}
  var acc = 0.0;
  ${swiglu ? "var acc2 = 0.0;" : ""}
  if (n < p.N) {
    let wrow = wbase + n * K8;
    let xb = (xrow * p.xStride) / 4u;
    for (var k = lane; k < K8; k += 32u) {
      let a = x[xb + 2u * k];
      let b = x[xb + 2u * k + 1u];
      acc += dot8(W[wrow + k], a, b);
      ${swiglu ? "acc2 += dot8(W2[wrow + k], a, b);" : ""}
    }
  }
  red[l.x] = acc;
  ${swiglu ? "red2[l.x] = acc2;" : ""}
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[l.x] += red[l.x + s];
      ${swiglu ? "red2[l.x] += red2[l.x + s];" : ""}
    }
    workgroupBarrier();
  }
  if (lane == 0u && n < p.N) {
    let o = m * p.yStride + p.nOffset + n;
    ${swiglu ? "let g = red[l.x]; let v = (g / (1.0 + exp(-g))) * red2[l.x];" : "let v = red[l.x];"}
    ${accumulate ? "y[o] = y[o] + v;" : "y[o] = v;"}
  }
}
`;
}

/**
 * Tiled y[m, n] (+)= sum_k x[m, k] * W[n, k] for prefill. 64x64 output tile per
 * workgroup, 16x16 threads, 4x4 outputs per thread, K step 16.
 *
 * grouped: rows are grouped by expert (z = expert). Row r of the group maps to
 * routing row list[offset[e] + r], whose input row is list / topk.
 */
export function gemmShader(opts: { accumulate: boolean; grouped: boolean; swiglu: boolean }): string {
  const { accumulate, grouped, swiglu } = opts;
  return /* wgsl */ `
${BF16}
struct P { M: u32, N: u32, K: u32, xStride: u32, yStride: u32, nOffset: u32, topk: u32, expertStride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${swiglu ? "@group(0) @binding(4) var<storage, read> W2: array<u32>;" : ""}
${grouped ? "@group(0) @binding(5) var<storage, read> groups: array<u32>;" : ""}
const BM = 64u;
const BN = 64u;
const BK = 16u;
var<workgroup> xs: array<f32, 1024>; // [BK][BM]
var<workgroup> ws: array<f32, 1024>; // [BK][BN]
${swiglu ? "var<workgroup> ws2: array<f32, 1024>;" : ""}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let tid = l.y * 16u + l.x;
  let n0 = wg.x * BN;
  let m0 = wg.y * BM;
  ${grouped ? `
  let e = wg.z;
  let count = groups[e * 2u + 1u];
  if (m0 >= count) { return; }
  let goff = groups[e * 2u];
  let listBase = 32u;
  let rows = min(BM, count - m0);
  let wbase = e * p.expertStride;
  ` : `
  let rows = min(BM, p.M - m0);
  let wbase = 0u;
  `}
  var acc: array<f32, 16>;
  ${swiglu ? "var acc2: array<f32, 16>;" : ""}
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; ${swiglu ? "acc2[i] = 0.0;" : ""} }
  // loader mapping: 256 threads load 64 rows x 16 k = 1024 values, 4 each
  let lrow = tid / 4u;          // 0..63
  let lk = (tid % 4u) * 4u;     // 0,4,8,12
  ${grouped ? `
  var xrow = 0u;
  if (lrow < rows) { xrow = groups[listBase + goff + m0 + lrow] / p.topk; }
  ` : "let xrow = m0 + lrow;"}
  let wn = n0 + lrow;
  let Kh = p.K / 2u;
  for (var k0 = 0u; k0 < p.K; k0 += BK) {
    if (lrow < rows) {
      let xb = xrow * p.xStride + k0 + lk;
      xs[(lk + 0u) * BM + lrow] = x[xb];
      xs[(lk + 1u) * BM + lrow] = x[xb + 1u];
      xs[(lk + 2u) * BM + lrow] = x[xb + 2u];
      xs[(lk + 3u) * BM + lrow] = x[xb + 3u];
    } else {
      xs[(lk + 0u) * BM + lrow] = 0.0;
      xs[(lk + 1u) * BM + lrow] = 0.0;
      xs[(lk + 2u) * BM + lrow] = 0.0;
      xs[(lk + 3u) * BM + lrow] = 0.0;
    }
    if (wn < p.N) {
      let wb = wbase + wn * Kh + (k0 + lk) / 2u;
      let a = W[wb];
      let b = W[wb + 1u];
      ws[(lk + 0u) * BN + lrow] = bf_lo(a);
      ws[(lk + 1u) * BN + lrow] = bf_hi(a);
      ws[(lk + 2u) * BN + lrow] = bf_lo(b);
      ws[(lk + 3u) * BN + lrow] = bf_hi(b);
      ${swiglu ? `
      let a2 = W2[wb];
      let b2 = W2[wb + 1u];
      ws2[(lk + 0u) * BN + lrow] = bf_lo(a2);
      ws2[(lk + 1u) * BN + lrow] = bf_hi(a2);
      ws2[(lk + 2u) * BN + lrow] = bf_lo(b2);
      ws2[(lk + 3u) * BN + lrow] = bf_hi(b2);
      ` : ""}
    } else {
      ws[(lk + 0u) * BN + lrow] = 0.0;
      ws[(lk + 1u) * BN + lrow] = 0.0;
      ws[(lk + 2u) * BN + lrow] = 0.0;
      ws[(lk + 3u) * BN + lrow] = 0.0;
      ${swiglu ? `
      ws2[(lk + 0u) * BN + lrow] = 0.0;
      ws2[(lk + 1u) * BN + lrow] = 0.0;
      ws2[(lk + 2u) * BN + lrow] = 0.0;
      ws2[(lk + 3u) * BN + lrow] = 0.0;
      ` : ""}
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let xv = vec4<f32>(xs[kk * BM + l.y * 4u], xs[kk * BM + l.y * 4u + 1u], xs[kk * BM + l.y * 4u + 2u], xs[kk * BM + l.y * 4u + 3u]);
      let wv = vec4<f32>(ws[kk * BN + l.x * 4u], ws[kk * BN + l.x * 4u + 1u], ws[kk * BN + l.x * 4u + 2u], ws[kk * BN + l.x * 4u + 3u]);
      ${swiglu ? "let wv2 = vec4<f32>(ws2[kk * BN + l.x * 4u], ws2[kk * BN + l.x * 4u + 1u], ws2[kk * BN + l.x * 4u + 2u], ws2[kk * BN + l.x * 4u + 3u]);" : ""}
      for (var i = 0u; i < 4u; i++) {
        let xi = xv[i];
        acc[i * 4u + 0u] += xi * wv.x;
        acc[i * 4u + 1u] += xi * wv.y;
        acc[i * 4u + 2u] += xi * wv.z;
        acc[i * 4u + 3u] += xi * wv.w;
        ${swiglu ? `
        acc2[i * 4u + 0u] += xi * wv2.x;
        acc2[i * 4u + 1u] += xi * wv2.y;
        acc2[i * 4u + 2u] += xi * wv2.z;
        acc2[i * 4u + 3u] += xi * wv2.w;
        ` : ""}
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let mr = l.y * 4u + i;
    if (mr >= rows) { continue; }
    ${grouped ? "let orow = groups[listBase + goff + m0 + mr];" : "let orow = m0 + mr;"}
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + l.x * 4u + j;
      if (n >= p.N) { continue; }
      let o = orow * p.yStride + p.nOffset + n;
      ${swiglu ? "let g = acc[i * 4u + j]; let v = (g / (1.0 + exp(-g))) * acc2[i * 4u + j];" : "let v = acc[i * 4u + j];"}
      ${accumulate ? "y[o] = y[o] + v;" : "y[o] = v;"}
    }
  }
}
`;
}

/**
 * Applies rotary embeddings to q (in place) and k, and writes k and v into the
 * KV cache. qkv rows hold [q (nH*D) | k (nKV*D) | v (nKV*D)].
 * One thread per (token, head, pair index).
 */
export const ropeCacheShader = /* wgsl */ `
struct P { T: u32, nH: u32, nKV: u32, D: u32, maxSeq: u32, qkvStride: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> rope: array<f32>; // [maxSeq][D/2] cos, then [maxSeq][D/2] sin
@group(0) @binding(3) var<storage, read_write> kc: array<f32>;
@group(0) @binding(4) var<storage, read_write> vc: array<f32>;
@group(0) @binding(5) var<storage, read> st: array<u32>; // st[0] = position of the first token
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let half = p.D / 2u;
  let heads = p.nH + 2u * p.nKV;
  let total = p.T * heads * half;
  let i = g.x;
  if (i >= total) { return; }
  let j = i % half;
  let h = (i / half) % heads;
  let t = i / (half * heads);
  let pos = st[0] + t;
  let base = t * p.qkvStride + h * p.D;
  let a = qkv[base + j];
  let b = qkv[base + j + half];
  if (h < p.nH + p.nKV) {
    let c = rope[pos * half + j];
    let s = rope[p.maxSeq * half + pos * half + j];
    let ra = a * c - b * s;
    let rb = b * c + a * s;
    if (h < p.nH) {
      qkv[base + j] = ra;
      qkv[base + j + half] = rb;
    } else {
      let kh = h - p.nH;
      let cb = (kh * p.maxSeq + pos) * p.D;
      kc[cb + j] = ra;
      kc[cb + j + half] = rb;
    }
  } else {
    let vh = h - p.nH - p.nKV;
    let cb = (vh * p.maxSeq + pos) * p.D;
    vc[cb + j] = a;
    vc[cb + j + half] = b;
  }
}
`;

/**
 * Causal attention with GQA. One workgroup per (query token, head); scores for
 * all keys are kept in workgroup memory (maxSeq <= 2048). D is the head dim
 * (a divisor of 256: 64 or 128).
 */
export function attentionShader(D: number): string {
  if (256 % D !== 0) throw new Error(`unsupported head_dim ${D}`);
  const G = 256 / D; // thread groups that split the keys in the weighted sum
  const sum = Array.from({ length: G }, (_, g) => `red[l.x + ${g * D}u]`).join(" + ");
  return /* wgsl */ `
const D = ${D}u;
struct P { T: u32, nH: u32, nKV: u32, D: u32, maxSeq: u32, qkvStride: u32, outStride: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> st: array<u32>;
var<workgroup> sc: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> qs: array<f32, ${D}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let h = wg.x;
  let t = wg.y;
  let kvh = h / (p.nH / p.nKV);
  let L = st[0] + t + 1u; // number of visible keys
  if (l.x < D) { qs[l.x] = qkv[t * p.qkvStride + h * D + l.x]; }
  workgroupBarrier();
  let kbase = kvh * p.maxSeq * D;
  var mx = -3.4e38;
  for (var s = l.x; s < L; s += 256u) {
    var d = 0.0;
    let kb = kbase + s * D;
    for (var i = 0u; i < D; i++) { d += qs[i] * kc[kb + i]; }
    d *= p.scale;
    sc[s] = d;
    mx = max(mx, d);
  }
  red[l.x] = mx;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l.x < k) { red[l.x] = max(red[l.x], red[l.x + k]); }
    workgroupBarrier();
  }
  let gmax = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var s = l.x; s < L; s += 256u) {
    let e = exp(sc[s] - gmax);
    sc[s] = e;
    sum += e;
  }
  red[l.x] = sum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l.x < k) { red[l.x] += red[l.x + k]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  workgroupBarrier();
  // ${G} groups of D threads split the keys; each thread owns one output dim.
  let d = l.x % D;
  let grp = l.x / D;
  var acc = 0.0;
  for (var s = grp; s < L; s += ${G}u) {
    acc += sc[s] * vc[kbase + s * D + d];
  }
  red[l.x] = acc;
  workgroupBarrier();
  if (l.x < D) {
    let v = (${sum}) * inv;
    out[t * p.outStride + h * D + l.x] = v;
  }
}
`;
}

/** Top-k routing: softmax over the k largest router logits (Megatron pre_softmax=False). */
export const topkShader = /* wgsl */ `
struct P { T: u32, E: u32, topk: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= p.T) { return; }
  var used = 0u; // bitmask, E <= 32
  var vals: array<f32, 8>;
  for (var k = 0u; k < p.topk; k++) {
    var best = 0u;
    var bv = -3.4e38;
    var found = false;
    for (var e = 0u; e < p.E; e++) {
      if ((used & (1u << e)) != 0u) { continue; }
      let v = logits[t * p.E + e];
      if (!found || v > bv) { bv = v; best = e; found = true; }
    }
    used |= 1u << best;
    sel[t * p.topk + k] = best;
    vals[k] = bv;
  }
  var s = 0.0;
  for (var k = 0u; k < p.topk; k++) { s += exp(vals[k] - vals[0]); }
  for (var k = 0u; k < p.topk; k++) { weights[t * p.topk + k] = exp(vals[k] - vals[0]) / s; }
}
`;

/**
 * Groups routing rows (t * topk + k) by expert. Single workgroup.
 * Output layout: [E][2] = (offset, count), padded to 32 u32, then the row list.
 */
export const moeGroupShader = /* wgsl */ `
struct P { R: u32, E: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read_write> groups: array<u32>;
var<workgroup> counts: array<atomic<u32>, 32>;
var<workgroup> offs: array<u32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  if (l.x < 32u) { atomicStore(&counts[l.x], 0u); }
  workgroupBarrier();
  for (var r = l.x; r < p.R; r += 256u) { atomicAdd(&counts[sel[r]], 1u); }
  workgroupBarrier();
  if (l.x == 0u) {
    var o = 0u;
    for (var e = 0u; e < p.E; e++) {
      let c = atomicLoad(&counts[e]);
      offs[e] = o;
      groups[e * 2u] = o;
      groups[e * 2u + 1u] = c;
      o += c;
    }
  }
  workgroupBarrier();
  // Deterministic placement: each row counts earlier rows with the same expert.
  for (var r = l.x; r < p.R; r += 256u) {
    let e = sel[r];
    var rank = 0u;
    for (var q = 0u; q < r; q++) { if (sel[q] == e) { rank++; } }
    groups[32u + offs[e] + rank] = r;
  }
}
`;

/** x[t] += sum_k w[t,k] * y[t*topk + k]  (residual add of the routed experts). */
export const moeCombineShader = /* wgsl */ `
struct P { T: u32, H: u32, topk: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.T * p.H) { return; }
  let t = i / p.H;
  let d = i % p.H;
  var s = 0.0;
  for (var k = 0u; k < p.topk; k++) {
    let r = t * p.topk + k;
    s += weights[r] * y[r * p.H + d];
  }
  x[i] = x[i] + s;
}
`;
