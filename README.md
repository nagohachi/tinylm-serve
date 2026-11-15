# tiny-lm chat

日本語の小さな言語モデル 2 つを、ブラウザの中だけで動かすチャットサイトです。サーバは不要で、GitHub Pages に静的ファイルとしてデプロイできます。

| モデル | 構成 | 重み (bf16) |
| --- | --- | --- |
| [nagohachi/tiny-lm-v3-japanese-1.5b-a500m-sft](https://huggingface.co/nagohachi/tiny-lm-v3-japanese-1.5b-a500m-sft) | Llama attention + MoE（16 experts から top-3、shared expert 1 個） | 2.92 GB |
| [nagohachi/tiny-lm-v1-japanese-500m-sft](https://huggingface.co/nagohachi/tiny-lm-v1-japanese-500m-sft) | Llama 型 dense | 0.96 GB |

## 仕組み

- 両モデルとも `trust_remote_code` のカスタム実装で、ONNX 版はありません。そのため、推論エンジンを WebGPU（WGSL）で書いています（`src/engine/`）。
- 重みは Hugging Face の `model.safetensors` を Range リクエストで 32 MB ずつ取得し、bf16 のまま GPU バッファに書き込みます。変換済みファイルを別にホストする必要はありません。
- 取得したチャンクは Cache API に保存するので、2 回目以降はダウンロードしません。キャッシュは重みファイルの LFS ハッシュ単位なので、README だけの更新ではキャッシュは無効になりません。古い重みのキャッシュは、新しい重みの読み込みに成功した後で削除します。オフラインでも、前回読み込んだリビジョンをキャッシュから使えます。
- 20 秒間データが届かないチャンクは中断して取り直します（最大 6 回）。
- 行列演算は bf16 の重みを f32 に展開して計算し、活性値と KV キャッシュは f32 です。prefill はタイル化した GEMM、decode は GEMV を使います。MoE の prefill ではトークンを expert ごとにまとめて計算します。
- トークナイザ（Unigram + byte fallback）は TypeScript で実装しています（`src/engine/tokenizer.ts`）。`@huggingface/tokenizers` は改行や一部の絵文字を `<unk>` にしてしまい、HF と結果が一致しなかったためです。
- 推論は Web Worker で動かし、UI をブロックしません。会話が続く場合は、前回までの KV キャッシュのうちプロンプトと共通する部分を再利用します。

## 動作環境

WebGPU が使えるブラウザが必要です（デスクトップ版 Chrome / Edge、WebGPU 対応版の Safari・Firefox）。GPU メモリは v1 で約 1.2 GB、v3 で約 3.2 GB 使います。

## 開発

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # dist/ に出力
```

### テスト

PyTorch（transformers）の出力と比べて、エンジンとトークナイザが一致するかを確認します。Node 上の WebGPU（[`webgpu`](https://www.npmjs.com/package/webgpu) パッケージ、Dawn）で動きます。

```sh
# 参照出力の生成（Python 環境に torch, transformers>=5, tokenizers が必要）
python tests/reference/tokenizer_ref.py tokenizer.json tok_ref.json
python tests/reference/chat_ref.py chat_ref.json
python tests/reference/model_ref.py <モデルのディレクトリ> ref_v1   # ref_v1.json と ref_v1_case*_last.f32 を出力

# 比較
node tests/tokenizer.test.ts tokenizer.json tok_ref.json chat_ref.json
node tests/model.test.ts <モデルのディレクトリ> ref_v1
```

モデルのディレクトリには `config.json`、`tokenizer.json`、`model.safetensors` と、モデルのリポジトリにある Python ファイルを置きます。

手元（Apple M4）での結果:

- トークナイザ: ランダム文字列・デコード・チャットテンプレートの 719 件がすべて一致
- 最終位置のロジットの最大誤差: v1 で 2.7e-5、v3 で 1.9e-5（ロジットの最大値は約 20）
- greedy 生成: 3 ケース（466 トークンの複数ターン会話を含む）すべてで PyTorch と同じトークン列
- 生成速度: 約 40 tok/s（Chrome、v1 / v3 とも）

## GitHub Pages へのデプロイ

`.github/workflows/deploy.yml` が `main` への push ごとにビルドしてデプロイします。初回だけ、リポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にしてください。Vite の `base` は `./` なので、`https://<user>.github.io/<repo>/` のどのパスでも動きます。
