import type { SamplingParams } from "./sampler.ts";
import type { ChatMessage } from "./tokenizer.ts";

export type WorkerRequest =
  | { type: "load"; modelId: string; repo: string; fallbackBytes: number }
  | { type: "generate"; id: number; messages: ChatMessage[]; params: SamplingParams }
  | { type: "stop" }
  | { type: "unload" }
  | { type: "cacheStatus"; repos: string[] }
  /** Deletes cached files of one repo, or everything when `repo` is omitted. */
  | { type: "clearCache"; repo?: string };

export interface GenerationStats {
  promptTokens: number;
  reusedTokens: number;
  newTokens: number;
  prefillMs: number;
  decodeMs: number;
  stopReason: "eos" | "length" | "context" | "stopped" | "stop-sequence";
  droppedMessages: number;
}

export type WorkerResponse =
  | { type: "status"; text: string }
  | {
      type: "progress";
      doneBytes: number;
      totalBytes: number;
      fromCache: boolean;
    }
  | { type: "ready"; modelId: string; adapter: string; contextLength: number }
  | { type: "loadError"; message: string }
  | { type: "token"; id: number; text: string }
  | { type: "prefill"; id: number; done: number; total: number }
  | { type: "done"; id: number; text: string; stats: GenerationStats }
  | { type: "genError"; id: number; message: string }
  | { type: "cacheStatus"; cached: Record<string, { bytes: number; complete: boolean }>; usage: number; quota: number }
  | { type: "cleared" };
