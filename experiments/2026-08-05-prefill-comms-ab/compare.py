"""Compare the prefill A/B/C/D arms: TTFT, collective cost, and protocol.

TTFT on a 16K prompt with max_tokens=1 is the prefill, so it is the headline.
The traces answer *why*: which NCCL protocol the kernels actually used and what
bus rate resulted. A timing delta without the kernel-name change would mean
something other than the protocol moved.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
SMEM = HERE.parent / "2026-07-29-marlin-smem-monopoly"
sys.path.insert(0, str(SMEM))

from analyze_step_budget import GPU_CATEGORIES, load_trace, step_windows  # noqa: E402

ARMS = ["baseline", "proto-simple", "chunk16k", "chunk16k-simple"]
HIDDEN = 6144


def ttft(arm: str) -> dict | None:
    vals = []
    for path in sorted(HERE.glob(f"{arm}-ttft-r*.json")):
        d = json.loads(path.read_text())
        vals.append(d["mean_ttft_ms"] / 1000)
    if not vals:
        return None
    return {
        "reps": len(vals),
        "mean_s": statistics.fmean(vals),
        "spread_pct": (max(vals) - min(vals)) / statistics.fmean(vals) * 100,
    }


def trace_stats(root: Path, arm: str) -> dict | None:
    d = root / arm
    traces = sorted(d.glob("*rank0*.trace.json.gz")) if d.is_dir() else []
    if not traces:
        return None
    events = load_trace(traces[0])
    windows = [w for w in step_windows(events) if w[1] - w[0] >= 200]
    if not windows:
        return None
    inside = lambda t: any(a <= t < b for a, b in windows)  # noqa: E731
    per_kernel: dict[str, float] = collections.defaultdict(float)
    counts: collections.Counter = collections.Counter()
    for e in events:
        if e.get("cat") not in GPU_CATEGORIES or not inside(e["t"]):
            continue
        if "nccl" in e["name"] or "cross_device_reduce" in e["name"]:
            key = e["name"].split("(")[0]
            per_kernel[key] += e["dur"] / 1000
            counts[key] += 1
    chunks = len(windows)
    comm_ms = sum(per_kernel.values()) / chunks
    ar = [(k, v) for k, v in per_kernel.items() if "AllReduce" in k]
    ar_ms = sum(v for _, v in ar) / chunks
    ar_n = sum(counts[k] for k, _ in ar) / chunks
    proto = "SIMPLE" if any("SIMPLE" in k for k in per_kernel) else (
        "LL128" if any("LL128" in k for k in per_kernel) else
        "LL" if any("_LL" in k for k in per_kernel) else "?"
    )
    return {
        "chunks": chunks,
        "chunk_wall_ms": sum(b - a for a, b in windows) / chunks,
        "protocol": proto,
        "comm_ms": comm_ms,
        "allreduce_ms": ar_ms,
        "allreduce_calls": ar_n,
        "kernels": {k: round(v / chunks, 1) for k, v in
                    sorted(per_kernel.items(), key=lambda kv: -kv[1])[:4]},
    }


def main() -> None:
    root = Path(sys.argv[1])
    rows = {}
    for arm in ARMS:
        rows[arm] = {"ttft": ttft(arm), "trace": trace_stats(root, arm)}

    base = rows["baseline"]["ttft"]
    print(f"{'arm':18} {'TTFT s':>9} {'vs base':>9} {'spread':>7} "
          f"{'proto':>7} {'comm ms':>9} {'AR ms':>8} {'AR n':>6}")
    for arm in ARMS:
        t, tr = rows[arm]["ttft"], rows[arm]["trace"]
        if t is None:
            print(f"{arm:18} {'(no result)':>9}")
            continue
        rel = f"{base['mean_s'] / t['mean_s']:.3f}x" if base else "-"
        if tr:
            print(f"{arm:18} {t['mean_s']:9.3f} {rel:>9} {t['spread_pct']:6.2f}% "
                  f"{tr['protocol']:>7} {tr['comm_ms']:9.1f} "
                  f"{tr['allreduce_ms']:8.1f} {tr['allreduce_calls']:6.0f}")
        else:
            print(f"{arm:18} {t['mean_s']:9.3f} {rel:>9} {t['spread_pct']:6.2f}% "
                  f"{'(no trace)':>7}")

    for arm in ARMS:
        tr = rows[arm]["trace"]
        if tr:
            print(f"\n{arm}: {tr['chunks']} chunk(s) of {tr['chunk_wall_ms']:.0f} ms")
            for k, v in tr["kernels"].items():
                print(f"    {v:8.1f} ms  {k[:72]}")

    (HERE / "comparison.json").write_text(
        json.dumps(rows, indent=2, default=float) + "\n"
    )


if __name__ == "__main__":
    main()
