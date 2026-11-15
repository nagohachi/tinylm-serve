import json, sys, random
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
random.seed(0)
pools = ["あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
 "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー",
 "日本語東京都首都秋夜詩書自己紹介説明指示応答機械学習言語模型計算", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
 " \n\t\r", "、。！？「」（）・…—~!@#$%^&*()_+-=[]{};':\",./<>?\\|`", "😀🎉👍🏻❤️🇯🇵𠮷①㈱ｶﾅ½é", "　​﻿"]
texts = []
for i in range(400):
    n = random.randint(1, 60)
    s = "".join(random.choice(random.choice(pools)) for _ in range(n))
    texts.append(s)
texts += ["", " ", "\n", "<s>", "a<s>b</s>c", "<unk><MASK|LLM-jp>x", "<0x0A>", "▁hello", "これは テスト です。\n\n### 指示:\n本文"]
enc = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts]
dec = []
for i in range(300):
    ids = [random.randrange(0, 99574) for _ in range(random.randint(1, 30))]
    if i % 3 == 0:
        ids = [random.randrange(8, 264) for _ in range(random.randint(1, 12))]
    dec.append({"ids": ids, "text": tok.decode(ids, skip_special_tokens=True)})
json.dump({"encode": enc, "decode": dec}, open(sys.argv[2], "w"), ensure_ascii=False)
print(len(enc), len(dec))
