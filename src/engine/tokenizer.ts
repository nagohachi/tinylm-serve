/**
 * Unigram tokenizer with byte fallback, matching the Hugging Face `tokenizers`
 * behaviour for the llm-jp-3 tokenizer used by the tiny-lm models.
 *
 * Pipeline (from tokenizer.json):
 *   added tokens are split out first (special, not normalized)
 *   normalizer: prepend "▁" at the start of each segment, replace " " with "▁"
 *   model: Unigram (Viterbi, fused unk, byte fallback)
 *   decoder: ByteFallback -> "▁" to " " -> Fuse -> strip one leading space
 */

interface TokenizerJson {
  added_tokens: { id: number; content: string; special: boolean }[];
  model: { type: string; unk_id: number; byte_fallback: boolean; vocab: [string, number][] };
}

const UNK_PENALTY = 10.0;
const SPACE_MARK = "▁";

export class UnigramTokenizer {
  readonly vocabSize: number;
  readonly bosId: number;
  readonly eosId: number;
  private readonly pieces: string[];
  private readonly scores: Float64Array;
  private readonly pieceToId = new Map<string, number>();
  private readonly unkId: number;
  private readonly minScore: number;
  private readonly maxPieceChars: number;
  private readonly byteFallback: boolean;
  private readonly byteIds = new Int32Array(256).fill(-1);
  private readonly idToByte: Int16Array;
  private readonly specialIds = new Set<number>();
  private readonly addedRegex: RegExp | null;
  private readonly addedToId = new Map<string, number>();
  private readonly utf8 = new TextEncoder();

  constructor(json: TokenizerJson) {
    if (json.model.type !== "Unigram") throw new Error(`Unsupported tokenizer model: ${json.model.type}`);
    const vocab = json.model.vocab;
    this.vocabSize = vocab.length;
    this.pieces = new Array(vocab.length);
    this.scores = new Float64Array(vocab.length);
    this.idToByte = new Int16Array(vocab.length).fill(-1);
    let minScore = Infinity;
    let maxChars = 1;
    for (let i = 0; i < vocab.length; i++) {
      const [piece, score] = vocab[i];
      this.pieces[i] = piece;
      this.scores[i] = score;
      // Rust inserts with HashMap::insert, so a duplicated piece maps to its last id.
      this.pieceToId.set(piece, i);
      if (score < minScore) minScore = score;
      let n = 0;
      for (const _ of piece) n++;
      if (n > maxChars) maxChars = n;
      const m = /^<0x([0-9A-F]{2})>$/.exec(piece);
      if (m) {
        const b = parseInt(m[1], 16);
        this.byteIds[b] = i;
        this.idToByte[i] = b;
      }
    }
    this.minScore = minScore;
    this.maxPieceChars = maxChars;
    this.unkId = json.model.unk_id;
    this.byteFallback = json.model.byte_fallback;
    for (const t of json.added_tokens) {
      this.addedToId.set(t.content, t.id);
      if (t.special) this.specialIds.add(t.id);
    }
    // Longest first, so that overlapping added tokens prefer the longer match.
    const sorted = [...json.added_tokens].sort((a, b) => b.content.length - a.content.length);
    this.addedRegex = sorted.length
      ? new RegExp(sorted.map((t) => t.content.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g")
      : null;
    this.bosId = this.addedToId.get("<s>") ?? 1;
    this.eosId = this.addedToId.get("</s>") ?? 2;
  }

  /** Encode text; special tokens written literally (e.g. "<s>") are recognised. No BOS is added. */
  encode(text: string): number[] {
    const ids: number[] = [];
    let last = 0;
    if (this.addedRegex) {
      this.addedRegex.lastIndex = 0;
      for (let m = this.addedRegex.exec(text); m; m = this.addedRegex.exec(text)) {
        if (m.index > last) this.encodeSegment(text.slice(last, m.index), ids);
        ids.push(this.addedToId.get(m[0])!);
        last = m.index + m[0].length;
      }
    }
    if (last < text.length) this.encodeSegment(text.slice(last), ids);
    return ids;
  }

  private normalize(segment: string): string {
    return SPACE_MARK + segment.replace(/ /g, SPACE_MARK);
  }

  private encodeSegment(segment: string, out: number[]): void {
    const normalized = this.normalize(segment);
    for (const piece of this.viterbi(normalized)) {
      const id = this.pieceToId.get(piece);
      if (id !== undefined) {
        out.push(id);
        continue;
      }
      if (this.byteFallback) {
        const bytes = this.utf8.encode(piece);
        let ok = true;
        for (const b of bytes) if (this.byteIds[b] < 0) ok = false;
        if (ok) {
          for (const b of bytes) out.push(this.byteIds[b]);
          continue;
        }
      }
      out.push(this.unkId);
    }
  }

  /** Port of `Unigram::encode_optimized` from huggingface/tokenizers. */
  private viterbi(s: string): string[] {
    const chars = Array.from(s);
    const n = chars.length;
    const bestScore = new Float64Array(n + 1);
    const startsAt = new Int32Array(n + 1).fill(-1);
    const nodeId = new Int32Array(n + 1);
    const unkScore = this.minScore - UNK_PENALTY;
    for (let start = 0; start < n; start++) {
      const base = bestScore[start];
      let hasSingle = false;
      let piece = "";
      const maxLen = Math.min(this.maxPieceChars, n - start);
      for (let len = 1; len <= maxLen; len++) {
        piece += chars[start + len - 1];
        const id = this.pieceToId.get(piece);
        if (id === undefined) continue;
        const end = start + len;
        const cand = base + this.scores[id];
        if (startsAt[end] < 0 || cand > bestScore[end]) {
          bestScore[end] = cand;
          startsAt[end] = start;
          nodeId[end] = id;
        }
        if (len === 1) hasSingle = true;
      }
      if (!hasSingle) {
        const end = start + 1;
        const cand = base + unkScore;
        if (startsAt[end] < 0 || cand > bestScore[end]) {
          bestScore[end] = cand;
          startsAt[end] = start;
          nodeId[end] = this.unkId;
        }
      }
    }
    const results: string[] = [];
    let unkRun: string[] = [];
    for (let end = n; end > 0; ) {
      const start = startsAt[end];
      const piece = chars.slice(start, end).join("");
      if (nodeId[end] === this.unkId) {
        unkRun.push(piece);
      } else {
        if (unkRun.length) {
          results.push(unkRun.reverse().join(""));
          unkRun = [];
        }
        results.push(piece);
      }
      end = start;
    }
    if (unkRun.length) results.push(unkRun.reverse().join(""));
    return results.reverse();
  }

  /**
   * Decode ids to text (special tokens skipped), like `tokenizer.decode(ids, skip_special_tokens=True)`.
   * With `lenient`, invalid UTF-8 inside a run of byte tokens only replaces the bad
   * sequence (HF replaces every byte of the run), which suits streaming display.
   */
  decode(ids: ArrayLike<number>, opts: { lenient?: boolean } = {}): string {
    const parts: string[] = [];
    let bytes: number[] = [];
    const utf8 = new TextDecoder("utf-8", { fatal: !opts.lenient });
    const flush = () => {
      if (!bytes.length) return;
      try {
        parts.push(utf8.decode(new Uint8Array(bytes)));
      } catch {
        for (let i = 0; i < bytes.length; i++) parts.push("�");
      }
      bytes = [];
    };
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (this.specialIds.has(id)) continue;
      if (id < 0 || id >= this.vocabSize) continue;
      const b = this.idToByte[id];
      if (b >= 0) {
        bytes.push(b);
        continue;
      }
      flush();
      parts.push(this.pieces[id]);
    }
    flush();
    const text = parts.join("").replaceAll(SPACE_MARK, " ");
    return text.startsWith(" ") ? text.slice(1) : text;
  }

  isSpecial(id: number): boolean {
    return this.specialIds.has(id);
  }
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export const DEFAULT_SYSTEM_TEXT = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。";

/**
 * The llm-jp-3 instruct chat template shipped with both models.
 * A system message is rendered as the fixed instruction sentence, whatever its content.
 */
export function renderChatTemplate(messages: ChatMessage[], addGenerationPrompt = true): string {
  let s = "<s>";
  messages.forEach((m, i) => {
    if (m.role === "user") s += "\n\n### 指示:\n" + m.content;
    else if (m.role === "system") s += DEFAULT_SYSTEM_TEXT;
    else if (m.role === "assistant") s += "\n\n### 応答:\n" + m.content + "</s>";
    if (i === messages.length - 1 && addGenerationPrompt) s += "\n\n### 応答:\n";
  });
  return s;
}
