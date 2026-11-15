// Compares the TS tokenizer against reference outputs produced by HF `tokenizers`.
// Usage: node tests/tokenizer.test.ts <tokenizer.json> <ref.json> [chat_ref.json]
import fs from "node:fs";
import { UnigramTokenizer, renderChatTemplate } from "../src/engine/tokenizer.ts";

const [tjPath, refPath, chatRefPath] = process.argv.slice(2);
const tok = new UnigramTokenizer(JSON.parse(fs.readFileSync(tjPath, "utf8")));
const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
let ok = 0, bad = 0;
for (const r of ref.encode) {
  const ids = tok.encode(r.text);
  if (eq(ids, r.ids)) ok++;
  else if (bad++ < 5) console.log("ENCODE MISMATCH", JSON.stringify(r.text), "\n ts", ids, "\n py", r.ids);
}
for (const r of ref.decode) {
  const text = tok.decode(r.ids);
  if (text === r.text) ok++;
  else if (bad++ < 10) console.log("DECODE MISMATCH", r.ids, "\n ts", JSON.stringify(text), "\n py", JSON.stringify(r.text));
}
if (chatRefPath) {
  const chat = JSON.parse(fs.readFileSync(chatRefPath, "utf8"));
  for (const r of chat.chat) {
    const text = renderChatTemplate(r.messages, true);
    const ids = tok.encode(text);
    if (text === r.text && eq(ids, r.ids)) ok++;
    else if (bad++ < 10) console.log("CHAT MISMATCH", JSON.stringify(text), JSON.stringify(r.text), "\n ts", ids, "\n py", r.ids);
  }
  for (const r of chat.plain) {
    const ids = tok.encode(r.text);
    if (eq(ids, r.ids) && tok.decode(ids) === r.decoded) ok++;
    else if (bad++ < 10) console.log("PLAIN MISMATCH", JSON.stringify(r.text), "\n ts", ids, "\n py", r.ids);
  }
}
console.log({ ok, bad });
process.exit(bad ? 1 : 0);
