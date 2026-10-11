"""Tokenizer cost on distinct texts (no repeat of an identical string), and on
an agentic turn shape: the same 60K-token prefix with a new suffix each time.

    tok_bench2.py [--fastokens] --ids out.npz
"""
import argparse
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

tok = AutoTokenizer.from_pretrained("/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16")
corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                 for f in sorted(Path("vllm").rglob("*.py")))
out = {}
for i, chars in enumerate((30_000, 180_000, 300_000)):
    for j in range(3):
        text = corpus[(i * 3 + j) * 400_000:][:chars]
        t0 = time.perf_counter()
        ids = tok.encode(text, add_special_tokens=False)
        dt = time.perf_counter() - t0
        out[f"d{i}{j}"] = np.array(ids)
        print(f"distinct {len(ids):7d} tokens: {dt*1e3:7.1f} ms ({dt/len(ids)*1e6:.3f} us/tok)")
prefix = corpus[3_000_000:3_180_000]
for j in range(4):
    msgs = [{"role": "user", "content": prefix},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": corpus[4_000_000 + j * 5000:][:2000]}]
    t0 = time.perf_counter()
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    dt = time.perf_counter() - t0
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    out[f"t{j}"] = np.array(ids)
    print(f"turn {len(ids):7d} tokens (same prefix, new suffix): {dt*1e3:7.1f} ms")
np.savez(a.ids, **out)
