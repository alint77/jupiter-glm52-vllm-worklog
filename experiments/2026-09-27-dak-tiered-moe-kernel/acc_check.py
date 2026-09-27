#!/usr/bin/env python3
"""Accuracy of the tiered kernel's two paths on real MiMo expert weights.

    acc_check.py dump <dir> --layer 20 --expert 57   # writes codes/scales/x for w13 and w2
    (run `tiered_moe_sk accuracy w13 <dir>/w13` and `... w2 <dir>/w2` on a GPU)
    acc_check.py compare <dir>

Activations are heavy-tailed on purpose: per-channel log-normal magnitudes plus
a few outlier channels 50-200x the median, the regime that stresses a
fixed-precision activation format. The reference is fp64 over the bf16
activations and exactly dequantized weights. "Marlin numerics" is what the
production kernel computes: exact products, fp32 accumulation, bf16 output.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL = Path(f"/e/fscratch/profound/{os.environ['USER']}/models/MiMo-V2.6-Pro-RL")
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def tensor(name):
    idx = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
    with safe_open(MODEL / idx[name], "pt") as f:
        return f.get_tensor(name)


def codes_of(packed: torch.Tensor) -> np.ndarray:
    p = packed.view(torch.uint8).numpy()
    out = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = p >> 4
    return out


def activations(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    chan = torch.exp(torch.randn(k, generator=g) * 0.8)
    out_idx = torch.randperm(k, generator=g)[:12]
    chan[out_idx] *= torch.empty(12).uniform_(50, 200, generator=g)
    x = torch.randn((8, k), generator=g) * chan
    x *= torch.exp(torch.randn((8, 1), generator=g) * 1.5)  # tokens at different scales
    return x.to(torch.bfloat16)


def dump(args):
    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    gate, up, down = (f"{prefix}.{p}_proj" for p in ("gate", "up", "down"))
    mats = {
        "w13": (np.concatenate([codes_of(tensor(gate + ".weight")), codes_of(tensor(up + ".weight"))]),
                np.concatenate([tensor(gate + ".weight_scale").numpy(), tensor(up + ".weight_scale").numpy()])),
        "w2": (codes_of(tensor(down + ".weight")), tensor(down + ".weight_scale").numpy()),
    }
    for name, (codes, scales) in mats.items():
        d = Path(args.dir) / name
        d.mkdir(parents=True, exist_ok=True)
        codes.astype(np.uint8).tofile(d / "codes.bin")
        scales.astype(np.uint8).tofile(d / "scales.bin")
        x = activations(codes.shape[1], seed=1 if name == "w13" else 2)
        x.view(torch.int16).numpy().tofile(d / "x.bin")
        print(name, codes.shape, "scale exps", int(scales.min()) - 127, int(scales.max()) - 127)


def compare(args):
    for name in ("w13", "w2"):
        d = Path(args.dir) / name
        x = torch.from_numpy(np.fromfile(d / "x.bin", dtype=np.int16)).view(torch.bfloat16)
        scales = np.fromfile(d / "scales.bin", dtype=np.uint8)
        codes = np.fromfile(d / "codes.bin", dtype=np.uint8)
        n = scales.size * 32 // (x.numel() // 8)
        k = x.numel() // 8
        w = E2M1[codes.reshape(n, k)] * np.exp2(scales.reshape(n, k // 32).astype(np.float64) - 127).repeat(32, 1)
        xd = x.float().double().view(8, k).numpy()
        ref = xd @ w.T
        marlin = torch.from_numpy((x.float().view(8, k) @ torch.from_numpy(w.T).float()).numpy())
        rows = {"marlin numerics (fp32 acc, bf16 out)": marlin.bfloat16().double().numpy()}
        for path in ("mma", "gemv"):
            y = np.fromfile(d / f"y_{path}.bin", dtype=np.float32).reshape(8, n).astype(np.float64)
            rows[f"tiered {path} path, fp32 out"] = y
            rows[f"tiered {path} path, bf16 out"] = torch.from_numpy(y).bfloat16().double().numpy()
        bf16_ref = torch.from_numpy(ref).bfloat16().double().numpy()
        print(f"\n{name}: {n}x{k}, 8 tokens, error vs fp64 reference (per token, relative to its output RMS)")
        print(f"  {'':40s} {'rms rel':>10s} {'max rel':>10s} {'== bf16(ref)':>13s}")
        for label, y in rows.items():
            rms = np.sqrt((ref ** 2).mean(1, keepdims=True))
            err = np.abs(y - ref) / rms
            same = (y == bf16_ref).mean() if "bf16" in label else float("nan")
            print(f"  {label:40s} {np.sqrt((err ** 2).mean()):10.2e} {err.max():10.2e} {same:13.4f}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "compare"])
    ap.add_argument("dir")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--expert", type=int, default=57)
    args = ap.parse_args()
    dump(args) if args.mode == "dump" else compare(args)
