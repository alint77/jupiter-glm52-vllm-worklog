"""Per-call durations of the sliced MoE layer kernel (tiered_decode::layer_kernel)
inside verify graphs (cudaGraphLaunch correlation with >= 500 kernels), per
device, from an Nsight Systems report with graph kernels traced per node.

    nsys_moe_calls.py <report.sqlite | report.nsys-rep> ... [--json out]
"""
import argparse
import collections
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from nsys_bd import NSYS


def calls(path):
    p = Path(path)
    if p.suffix != ".sqlite":
        q = p.with_suffix(".sqlite")
        if not q.exists():
            subprocess.run([NSYS, "export", "--type", "sqlite", "-o", str(q), str(p)],
                           check=True, capture_output=True)
        p = q
    c = sqlite3.connect(p)
    names = dict(c.execute("select id, value from StringIds"))
    out = {}
    for (dev,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall():
        by_corr = collections.defaultdict(list)
        for s, e, corr, n in c.execute("select start, end, correlationId, demangledName from "
                                       "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (dev,)):
            by_corr[corr].append((s, e, names[n]))
        steps = [ks for ks in by_corr.values() if len(ks) >= 500]
        per = [[(e - s) / 1e3 for s, e, n in ks if "layer_kernel" in n] for ks in steps]
        out[dev] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for r in a.reports:
        for dev, per in calls(r).items():
            d = np.concatenate([np.array(x) for x in per if x])
            n = collections.Counter(len(x) for x in per).most_common(1)[0][0]
            res[f"{r}:{dev}"] = {"steps": len(per), "calls_per_step": n, "mean_us": float(d.mean()),
                                 "p10": float(np.percentile(d, 10)), "p50": float(np.percentile(d, 50)),
                                 "p90": float(np.percentile(d, 90)), "sum_ms_per_step": float(d.sum() / len(per) / 1e3)}
            v = res[f"{r}:{dev}"]
            print(f"{r} dev{dev}: {v['steps']} steps x {n} calls, mean {v['mean_us']:.1f} us "
                  f"(p10/50/90 {v['p10']:.1f}/{v['p50']:.1f}/{v['p90']:.1f}), {v['sum_ms_per_step']:.2f} ms/step")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
