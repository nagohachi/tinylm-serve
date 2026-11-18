# tiny-lm chat

3 つの日本語言語モデルを、ブラウザの中で動かすチャットサイトです。推論はすべて利用者の端末の GPU で行い、サーバは使いません。

https://nagohachi.github.io/tinylm-serve/

## モデル

| モデル | 初回のダウンロード量 | GPU メモリの使用量（目安） |
| --- | --- | --- |
| [tiny-lm-v3-japanese-1.5b-a500m-sft](https://huggingface.co/nagohachi/tiny-lm-v3-japanese-1.5b-a500m-sft) | 2.92 GB | 3.2 GB |
| [Llama-3.2-tiny-lm-v2-japanese-500m-sft](https://huggingface.co/nagohachi/Llama-3.2-tiny-lm-v2-japanese-500m-sft) | 0.95 GB | 1.2 GB |
| [tiny-lm-v1-japanese-500m-sft](https://huggingface.co/nagohachi/tiny-lm-v1-japanese-500m-sft) | 0.96 GB | 1.2 GB |

重みは Hugging Face からダウンロードしてブラウザに保存するので、2 回目以降はダウンロードしません。保存した重みは、設定画面かモデル選択画面のボタンでモデルごとに削除できます。

v2 は Meta の Llama 3.2 から作ったモデルで、[Llama 3.2 Community License](https://huggingface.co/nagohachi/Llama-3.2-tiny-lm-v2-japanese-500m-sft/blob/main/LICENSE) に従います（Built with Llama）。

## 必要な環境

WebGPU（ブラウザから GPU で計算するための API。[MDN の解説](https://developer.mozilla.org/ja/docs/Web/API/WebGPU_API)）に対応したブラウザが必要です。動作は macOS 版の Chrome で確認しています。

## 開発

```sh
npm install
npm run dev     # http://localhost:5173 で起動する
npm run build   # dist/ に静的ファイルを出力する
```

`main` ブランチに push すると、GitHub Actions がビルドして GitHub Pages にデプロイします。

推論結果を PyTorch の出力と比べるテストが `tests/` にあります。使い方は `tests/model.test.ts` と `tests/tokenizer.test.ts` の先頭に書いてあります。
