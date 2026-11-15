export interface SamplingParams {
  temperature: number;
  topP: number;
  /** 0 disables top-k. */
  topK: number;
  /** 1 disables the penalty. Applied to the tokens generated so far. */
  repetitionPenalty: number;
  maxNewTokens: number;
}

export const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.7,
  topP: 0.9,
  topK: 0,
  repetitionPenalty: 1.1,
  maxNewTokens: 512,
};

/**
 * Picks the next token. `allowed` is the number of real vocabulary entries
 * (the model's embedding can be padded beyond it); `banned` ids are never chosen.
 */
export function sampleToken(
  logitsIn: Float32Array,
  allowed: number,
  banned: readonly number[],
  p: SamplingParams,
  generated: readonly number[],
  rand: () => number = Math.random,
): number {
  const logits = logitsIn.subarray(0, allowed);
  for (const id of banned) if (id < allowed) logits[id] = -Infinity;
  if (p.repetitionPenalty !== 1) {
    for (const id of new Set(generated)) {
      if (id >= allowed) continue;
      const v = logits[id];
      logits[id] = v > 0 ? v / p.repetitionPenalty : v * p.repetitionPenalty;
    }
  }
  let best = 0;
  for (let i = 1; i < allowed; i++) if (logits[i] > logits[best]) best = i;
  if (p.temperature <= 0) return best;

  const invT = 1 / p.temperature;
  const max = logits[best];
  const probs = new Float32Array(allowed);
  let sum = 0;
  for (let i = 0; i < allowed; i++) {
    const e = Math.exp((logits[i] - max) * invT);
    probs[i] = e;
    sum += e;
  }
  // Candidates: everything with non-negligible probability, sorted by probability.
  const threshold = sum * 1e-7;
  let cand: number[] = [];
  for (let i = 0; i < allowed; i++) if (probs[i] > threshold) cand.push(i);
  cand.sort((a, b) => probs[b] - probs[a]);
  if (p.topK > 0 && cand.length > p.topK) cand = cand.slice(0, p.topK);
  let candSum = 0;
  for (const i of cand) candSum += probs[i];
  if (p.topP < 1) {
    let acc = 0;
    let keep = 0;
    for (; keep < cand.length; keep++) {
      acc += probs[cand[keep]] / candSum;
      if (acc >= p.topP) {
        keep++;
        break;
      }
    }
    cand = cand.slice(0, Math.max(1, keep));
    candSum = 0;
    for (const i of cand) candSum += probs[i];
  }
  let r = rand() * candSum;
  for (const i of cand) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return cand[cand.length - 1];
}
