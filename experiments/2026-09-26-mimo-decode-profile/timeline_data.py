#!/usr/bin/env python3
"""Median timeline of a MiMo-V2.6 decode step and of one MoE layer, from traces.

    timeline_data.py <trace-dir> --out timeline-2092055.json

The verify graph is one stream, so a layer is a fixed kernel sequence between
TP all-reduces: 141 per step = 1 after the embedding + 2 per layer x 70.
Layer 0 is the dense MLP; of the 69 MoE layers, the sliding-window ones share
one 20-kernel sequence, which is aligned by position across steps, layers and
ranks. The MoE-closing all-reduce is split into transfer (the cross-rank
minimum for that step and layer) and waiting for the slowest GPU.
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import RANK_RE, load, steps  # noqa: E402

# byte floors per call, us (sol.py); MoE split by bytes from the per-layer
# E[max(hot/HBM, cold/C2C)] = 89.3 us (w13 is 2/3 of an expert's bytes)
FLOORS = {"qkv_proj (fp8)": 11.5, "o_proj (bf16)": 13.9, "router gate (bf16)": 1.3,
          "MoE w13, one kernel": 59.5, "MoE w2, one kernel": 29.8}


# (substring of the kernel name, display name, kind: bw / lat / comm)
RULES = [
    ("rms_norm", "RMSNorm (+ residual add)", "lat"),
    ("scale_1x128", "fp8 activation quant", "lat"),
    ("fp8_gemm_kernel_swapAB<6784u", "qkv_proj (fp8)", "bw"),
    ("triton_poi_fused_2", "rope / qk norm (fused)", "lat"),
    ("reshape_and_cache", "KV cache write", "lat"),
    ("FlashAttentionForwardCombine", "attention combine", "lat"),
    ("FlashAttentionForwardSm90", "attention (sliding 128)", "lat"),
    ("nvjet_sm90_tst_64x8", "o_proj (bf16)", "bw"),
    ("nvjet_sm90_tss_64x8", "router gate (bf16)", "bw"),
    ("splitKreduce", "router gate split-K reduce", "lat"),
    ("grouped_topk", "top-8 routing", "lat"),
    ("_assign_kernel", "replica assign + align", "lat"),
    ("route_prep_kernel", "MoE route + x prep", "lat"),
    ("gemm_kernel<0>", "MoE w13, one kernel", "bw"),
    ("act_kernel", "MoE silu*up", "lat"),
    ("gemm_kernel<1>", "MoE w2, one kernel", "bw"),
    ("finalize_kernel", "MoE finalize", "lat"),
    ("cross_device_reduce", "TP all-reduce", "comm"),
]


def label(name: str) -> tuple[str, str]:
    """(display name, kind) for a kernel."""
    for key, text, kind in RULES:
        if key in name:
            return text, kind
    return name[:60], "lat"


def layers_of(ops):
    ars = [i for i, o in enumerate(ops) if "cross_device_reduce" in o["name"]]
    out = []
    for k in range(len(ars) // 2):
        out.append(ops[ars[2 * k] + 1: ars[2 * k + 2] + 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    per_rank = {}
    for path in sorted(args.trace_dir.glob("*.pt.trace.json.gz")):
        per_rank[int(RANK_RE.search(path.name).group(1))] = steps(load(path))
    ranks = sorted(per_rank)
    n_steps = min(len(v) for v in per_rank.values())

    candidates = collections.defaultdict(list)
    layer_kind = {}
    pos = collections.defaultdict(lambda: {"start": [], "dur": []})
    ar_wait, ar_min = [], []
    layer_span = collections.defaultdict(list)       # layer index -> [(start, end) rel. step]
    phase = collections.defaultdict(list)
    for s in range(n_steps):
        for r in ranks:
            row = per_rank[r][s]
            t0 = row["span"][0]
            for name, ops in row["phases"].items():
                if ops:
                    phase[name].append((min(o["t"] for o in ops) - t0, max(o["end"] for o in ops) - t0,
                                        sum(o["end"] - o["t"] for o in ops)))
            phase["period"].append((0, row["period"], row["period"]))
            ops = sorted(row["phases"]["target"], key=lambda e: e["t"])
            for k, layer in enumerate(layers_of(ops)):
                layer_span[k].append((layer[0]["t"] - t0, layer[-1]["end"] - t0))
                if s == 0 and r == ranks[0]:
                    names = [o["name"] for o in layer]
                    layer_kind[k] = ("dense" if k == 0 else
                                     "full attention MoE" if any("FlashAttnFwdSm90" in n or "flash::FlashAttn" in n
                                                                  for n in names) else "sliding MoE")
                key = tuple(label(o["name"])[0] for o in layer)
                if k > 0 and "attention (sliding 128)" in key:
                    candidates[key].append((s, k, r, layer))

    # the sliding-window MoE layers' common sequence (the last layer differs)
    seq_key = max(candidates, key=lambda key: len(candidates[key]))
    layer_ar = collections.defaultdict(dict)
    for s_idx, k, r, layer in candidates[seq_key]:
        base = layer[0]["t"]
        for i, o in enumerate(layer):
            pos[i]["start"].append((o["t"] - base) * 1000)
            pos[i]["dur"].append((o["end"] - o["t"]) * 1000)
        layer_ar[(s_idx, k)][r] = (layer[-1]["end"] - layer[-1]["t"]) * 1000
    for by_rank in layer_ar.values():
        if len(by_rank) == len(ranks):
            m = min(by_rank.values())
            ar_min.append(m)
            ar_wait.extend(v - m for v in by_rank.values())
    kind_of = {text: kind for _, text, kind in RULES}
    kernels = []
    for i, name in enumerate(seq_key):
        d = pos[i]["dur"]
        q = statistics.quantiles(d, n=10)
        # means lay the layer out (they add up to the step budget); medians
        # and the p10-p90 band describe the spread
        kernels.append({"name": name, "kind": kind_of.get(name, "lat"),
                        "start": statistics.fmean(pos[i]["start"]), "dur": statistics.fmean(d),
                        "median": statistics.median(d), "p10": q[0], "p90": q[-1],
                        "floor": FLOORS.get(name)})
    closing = kernels[-1]
    closing["transfer"] = statistics.median(ar_min)
    closing["wait"] = statistics.mean(ar_wait)

    spans = {k: (statistics.median(a for a, _ in v) * 1000, statistics.median(b for _, b in v) * 1000)
             for k, v in layer_span.items()}
    phases = {name: {"start": statistics.median(a for a, _, _ in v) * 1000,
                     "end": statistics.median(b for _, b, _ in v) * 1000,
                     "busy": statistics.median(c for _, _, c in v) * 1000}
              for name, v in phase.items()}
    out = {"trace": str(args.trace_dir), "steps": n_steps, "ranks": len(ranks),
           "layer_kernels": kernels, "layer_spans_us": spans, "layer_kind": layer_kind, "phases_us": phases,
           "layer_sample_count": len(pos[0]["dur"])}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v["start"], 1), round(v["dur"], 1)) for k, v in
                      zip([x["name"] for x in kernels], kernels)}, indent=0)[:3000])
    print("phases", {k: {kk: round(vv) for kk, vv in v.items()} for k, v in phases.items()})
    print("AR transfer", round(closing["transfer"], 1), "wait", round(closing["wait"], 1),
          "samples", out["layer_sample_count"])


if __name__ == "__main__":
    main()
