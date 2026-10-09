"""Lossless headroom in the W4A16 experts: empirical entropy of the int4
codes (order 0, and conditioned on the previous code along K) and of the
bf16 group scales, for sampled experts across layers.

    int4_entropy.py <model dir>
"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

M = sys.argv[1]
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


tot0 = np.zeros(16)
pair = np.zeros((16, 16))
sbits = []
for layer in (5, 30, 55, 77):
    for e in (3, 111, 200):
        for proj in ("gate_proj", "down_proj"):
            k = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            with safe_open(f"{M}/{idx[k + '.weight_packed']}", "pt") as f:
                w = f.get_tensor(k + ".weight_packed")
            with safe_open(f"{M}/{idx[k + '.weight_scale']}", "pt") as f:
                s = f.get_tensor(k + ".weight_scale")
            w = w.view(torch.int32).numpy().view(np.uint32)
            codes = np.stack([(w >> (4 * i)) & 0xF for i in range(8)], -1).reshape(w.shape[0], -1)
            tot0 += np.bincount(codes.ravel(), minlength=16)
            np.add.at(pair, (codes[:, :-1].ravel(), codes[:, 1:].ravel()), 1)
            sv = s.view(torch.int16).numpy().ravel()
            sbits.append(H(np.unique(sv, return_counts=True)[1].astype(float)))
h0 = H(tot0)
h1 = sum(pair[i].sum() / pair.sum() * H(pair[i]) for i in range(16))
print("code histogram:", np.round(tot0 / tot0.sum(), 4).tolist())
print(f"int4 codes: order-0 entropy {h0:.3f} bits, order-1 (previous code along K) {h1:.3f} bits"
      f" -> {1 - h0 / 4:.1%} / {1 - h1 / 4:.1%} smaller than 4 bits")
print(f"bf16 scales: entropy {np.mean(sbits):.2f} bits of 16 (per tensor, mean)")
print(f"scale share of expert bytes: 16 / (128 x 4) = {16 / 512:.1%}")
