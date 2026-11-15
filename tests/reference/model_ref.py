import json, sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
d, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.float32).eval()
cases = [
  [{"role":"user","content":"日本の首都はどこですか？"}],
  [{"role":"user","content":"短い詩を書いてください。テーマは秋の夜です。"}],
  [{"role":"system","content":""},{"role":"user","content":"次の文章を要約してください。\n\n" + "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。 " * 6},
   {"role":"assistant","content":"猫が自分の生まれについて語る話です。"},{"role":"user","content":"もう少し詳しく説明して。"}],
]
res = []
for i, c in enumerate(cases):
    ids = tok.apply_chat_template(c, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")["input_ids"]
    logits = model(ids).logits[0].float()
    top = torch.topk(logits, 5, dim=-1)
    np.asarray(logits[-1].numpy(), dtype=np.float32).tofile(f"{out}_case{i}_last.f32")
    gen = model.generate(ids, max_new_tokens=48, do_sample=False)
    new = gen[0, ids.shape[1]:].tolist()
    res.append({"messages": c, "input_ids": ids[0].tolist(),
                "top5_ids": top.indices.tolist(), "top5_vals": top.values.tolist(),
                "greedy": new, "greedy_text": tok.decode(new, skip_special_tokens=True)})
    print(i, res[-1]["greedy_text"])
json.dump(res, open(out + ".json", "w"), ensure_ascii=False)
