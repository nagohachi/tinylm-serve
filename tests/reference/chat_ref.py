import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("nagohachi/tiny-lm-v1-japanese-500m-sft")
texts = ["日本の首都はどこですか？", "Hello World", "  leading spaces\nnew line\n\n### 指示:\nテスト", "絵文字😀とか特殊記号①㈱、半角ｶﾅ", "\n\nstarts with newline", "a\tb  c   d", "<s>は特殊?</s>", "コードを書いて: def f(x):\n    return x*2"]
out = {"plain": [], "chat": []}
for t in texts:
    ids = tok(t, add_special_tokens=False)["input_ids"]
    out["plain"].append({"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=True)})
convs = [
  [{"role":"user","content":"日本の首都はどこですか？"}],
  [{"role":"system","content":"x"},{"role":"user","content":"自己紹介して"},{"role":"assistant","content":"私はAIです。"},{"role":"user","content":"もう一度\n詳しく"}],
]
for c in convs:
    s = tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
    ids = tok.apply_chat_template(c, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    out["chat"].append({"messages": c, "text": s, "ids": ids})
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(json.dumps(out["chat"][0], ensure_ascii=False))
