export interface ModelEntry {
  id: string;
  repo: string;
  name: string;
  summary: string;
  details: string[];
  /** Fallback size (bytes) of model.safetensors if the Hub API is unreachable. */
  weightBytes: number;
  /** Attribution the model's license asks us to display. */
  notice?: { text: string; href: string };
}

export const MODELS: ModelEntry[] = [
  {
    id: "v3-1.5b-a500m",
    repo: "nagohachi/tiny-lm-v3-japanese-1.5b-a500m-sft",
    name: "tiny-lm v3 1.5B-A500M",
    summary: "MoE（16 experts から top-3 + shared expert）",
    details: ["総パラメータ 1.46B / アクティブ 480M", "事前学習 10.5B トークン", "コンテキスト長 2048"],
    weightBytes: 2_922_915_008,
  },
  {
    id: "v2-500m",
    repo: "nagohachi/Llama-3.2-tiny-lm-v2-japanese-500m-sft",
    name: "tiny-lm v2 500M",
    summary: "Llama-3.2-3B を枝刈りした Llama 型 dense モデル",
    details: ["パラメータ 476M", "枝刈り後に約 10B トークンで追加の事前学習", "コンテキスト長 2048"],
    weightBytes: 952_756_472,
    notice: {
      text: "Built with Llama（Llama 3.2 Community License）",
      href: "https://huggingface.co/nagohachi/Llama-3.2-tiny-lm-v2-japanese-500m-sft/blob/main/LICENSE",
    },
  },
  {
    id: "v1-500m",
    repo: "nagohachi/tiny-lm-v1-japanese-500m-sft",
    name: "tiny-lm v1 500M",
    summary: "Llama 型 dense モデル",
    details: ["パラメータ 480M", "事前学習 約 10B トークン", "コンテキスト長 2048"],
    weightBytes: 959_027_432,
  },
];

export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${(n / 1e3).toFixed(0)} KB`;
}
