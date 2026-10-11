"""Tokenizer cost at agentic prompt sizes, in-process (no server): HF fast
tokenizer (Rust `tokenizers`) as vLLM loads it, optionally with the fastokens
patch (VLLM_USE_FASTOKENS path). Prints median ms per encode, us per token,
and writes the ids so two backends can be compared for identity.

    tok_bench.py [--fastokens] --ids out.npz
"""
import argparse
import statistics as st
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fastokens", action="store_true")
ap.add_argument("--ids", required=True)
a = ap.parse_args()
if a.fastokens:
    import fastokens
    fastokens.patch_transformers()
from transformers import AutoTokenizer  # noqa: E402

M = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"
tok = AutoTokenizer.from_pretrained(M)
print(type(tok).__name__, type(tok._tokenizer).__module__ if hasattr(tok, "_tokenizer") else "")
corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                 for f in sorted(Path("vllm").rglob("*.py")))
out = {}
for chars in (3_000, 30_000, 90_000, 180_000, 300_000):
    text = "Below is source code from a project.\n" + corpus[:chars]
    msgs = [{"role": "user", "content": text}]
    ts, tc = [], []
    for _ in range(7):
        t0 = time.perf_counter()
        ids = tok.encode(text, add_special_tokens=False)
        ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        cids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        tc.append(time.perf_counter() - t0)
    cids = cids["input_ids"] if isinstance(cids, dict) or hasattr(cids, "keys") else cids
    out[f"e{chars}"] = np.array(ids)
    out[f"c{chars}"] = np.array(cids)
    n = len(ids)
    print(f"{n:7d} tokens: encode {st.median(ts)*1e3:7.1f} ms ({st.median(ts)/n*1e6:.2f} us/tok), "
          f"chat template+encode {st.median(tc)*1e3:7.1f} ms", flush=True)
np.savez(a.ids, **out)
