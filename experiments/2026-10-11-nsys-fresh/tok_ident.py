"""Token-id identity, HF tokenizers vs fastokens, on every text file in the
repo (code, docs, json, yaml, unicode), chat-templated agentic turns with
tool-style content, and unicode stress strings. Prints mismatches.

    tok_ident.py
"""
import subprocess
import sys
from pathlib import Path

import fastokens
from transformers import AutoTokenizer

M = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"
hf = AutoTokenizer.from_pretrained(M)
fastokens.patch_transformers()
ft = AutoTokenizer.from_pretrained(M)
assert type(ft._tokenizer).__module__ != type(hf._tokenizer).__module__
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
texts = []
for f in files:
    p = Path(f)
    if p.suffix in (".py", ".md", ".json", ".yaml", ".yml", ".txt", ".cu", ".cuh", ".cpp",
                    ".h", ".sh", ".toml", ".jinja", ".rst", ".html", ".css", ".js", ".cfg"):
        try:
            texts.append(p.read_text())
        except Exception:
            pass
stress = ["héllo wörld ñ ç ß", "日本語のテキストと中文混合 한국어", "emoji 🎉🚀👨‍👩‍👧 test",
          "\t\t  spaces   \n\n\r\n tabs", "<|user|> literal special <|assistant|>",
          "\u200b zero width \ufeff bom", "math ∑∫√∞ ≠ ≤", "a" * 5000, " " * 3000,
          "\\x00\x01\x02 control", "مرحبا بالعالم", "Ελληνικά кириллица"]
texts += stress
bad = 0
n = 0
for t in texts:
    a, b = hf.encode(t, add_special_tokens=False), ft.encode(t, add_special_tokens=False)
    n += len(a)
    if a != b:
        bad += 1
        if bad <= 5:
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print("MISMATCH", repr(t[:60]), len(a), len(b), "first diff at", i)
for k in range(20):
    msgs = [{"role": "system", "content": texts[k * 7 % len(texts)][:4000]},
            {"role": "user", "content": texts[k * 13 % len(texts)][:20000]},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "read_file", "arguments": {"path": f"/tmp/x{k}.py", "lines": [1, 50]}}}]},
            {"role": "tool", "content": texts[k * 17 % len(texts)][:8000]}]
    a = hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    b = ft.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    a = a["input_ids"] if hasattr(a, "keys") else a
    b = b["input_ids"] if hasattr(b, "keys") else b
    n += len(a)
    if list(a) != list(b):
        bad += 1
        print("CHAT MISMATCH", k, len(a), len(b))
    dec_a, dec_b = hf.decode(a), ft.decode(b)
    if dec_a != dec_b:
        bad += 1
        print("DECODE MISMATCH", k)
print(f"{len(texts)} texts + 20 chat turns, {n} tokens, mismatches: {bad}")
sys.exit(1 if bad else 0)
