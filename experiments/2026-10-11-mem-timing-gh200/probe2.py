"""HBM gather granularity sweep with cold/hot row regimes (probe2).

For each access granularity g (bf16 elements, 2B each), measure effective
bandwidth of random-index row gathers over an 8 GiB span, with indices drawn
either uniformly (cold: row-miss dominated) or resampled from a small fixed
row set (hot: open-page hits dominate). The cold-vs-hot contrast at a fixed
granularity isolates the row-activation cost from sector-utilization waste.
"""

import argparse
import json
import os
import time

import torch

GiB = 1024**3
DST_BYTES = 512 * 1024**2
GRAN_ELEM = [1, 2, 4, 16, 64, 128, 512, 2048, 8192, 32768]  # 2B .. 64KiB
HOT_ROWS = 4096
REPS = 12


def bench(fn, bytes_moved, reps=REPS, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return bytes_moved * reps / dt / GiB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    elem = 2  # bf16
    src = torch.empty(4 * 2**30, dtype=torch.bfloat16, device=dev)  # 8 GiB
    res = {
        "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "device": torch.cuda.get_device_name(dev),
        "hot_rows": HOT_ROWS,
        "dst_bytes": DST_BYTES,
        "sweep": [],
    }

    for g in GRAN_ELEM:
        row_b = g * elem
        n_rows = src.numel() // g
        out_rows = DST_BYTES // row_b
        srcv = src.view(-1, g)
        cold_idx = torch.randint(0, n_rows, (out_rows,), device=dev)
        hot_base = torch.randint(0, n_rows, (HOT_ROWS,), device=dev)
        hot_idx = hot_base[torch.randint(0, HOT_ROWS, (out_rows,), device=dev)]
        dst = torch.empty(out_rows, g, dtype=torch.bfloat16, device=dev)
        moved = out_rows * row_b * 2
        entry = {
            "gran_bytes": row_b,
            "cold_gibs": round(
                bench(lambda: torch.index_select(srcv, 0, cold_idx, out=dst),
                      moved), 2),
            "hot_gibs": round(
                bench(lambda: torch.index_select(srcv, 0, hot_idx, out=dst),
                      moved), 2),
        }
        entry["hot_over_cold"] = round(entry["hot_gibs"] / entry["cold_gibs"], 3)
        res["sweep"].append(entry)
        del cold_idx, hot_base, hot_idx, dst

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nprobe2 sweep -> {args.out}")
    print(f"{'gran':>8s} {'cold':>10s} {'hot':>10s} {'hot/cold':>9s}")
    for e in res["sweep"]:
        print(f"{e['gran_bytes']:8d} {e['cold_gibs']:10.1f} "
              f"{e['hot_gibs']:10.1f} {e['hot_over_cold']:9.3f}")


if __name__ == "__main__":
    main()
