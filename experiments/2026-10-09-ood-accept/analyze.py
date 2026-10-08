"""Acceptance by prompt / domain / mode for the ood_arm.sh runs; the
DFlash2 k=3 - MTP3 gap at the same draft length; per-position rates.

    analyze.py [root]
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/e/fscratch/profound/naeimitabiei1/ood-accept")
ARMS = ["mtp3", "dflash2-k3", "dflash2-k7"]
data = {}
for arm in ARMS:
    f = root / arm / "acc.jsonl"
    if f.exists():
        data[arm] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def pooled(rows):
    """Acceptance length over all drafts of these requests (token-weighted)."""
    d = sum(r["drafts"] for r in rows)
    return 1 + sum(r["accepted"] for r in rows) / d if d else float("nan")


for mode in ("content", "thinking"):
    print(f"\n===== mode: {mode} (acceptance length, pooled over seeds)")
    ids = [r["id"] for r in data.get("mtp3", next(iter(data.values())))
           if r["mode"] == mode and r["seed"] == 0]
    dom = {r["id"]: r["domain"] for rows in data.values() for r in rows}
    print(f"{'prompt':15s} {'domain':6s} " + " ".join(f"{a:>11s}" for a in data)
          + "   k3 gap (DF2 - MTP3)")
    for pid in ids:
        vals = {a: pooled([r for r in rows if r["mode"] == mode and r["id"] == pid])
                for a, rows in data.items()}
        gap = vals.get("dflash2-k3", float("nan")) - vals.get("mtp3", float("nan"))
        print(f"{pid:15s} {dom[pid]:6s} " + " ".join(f"{vals[a]:11.2f}" for a in data)
              + f"   {gap:+.2f}")
    print("-- by domain")
    for d in ("code", "math", "prose", "other", "all"):
        vals = {a: pooled([r for r in rows if r["mode"] == mode and (d == "all" or r["domain"] == d)])
                for a, rows in data.items()}
        gap = vals.get("dflash2-k3", float("nan")) - vals.get("mtp3", float("nan"))
        print(f"{d:15s} {'':6s} " + " ".join(f"{vals[a]:11.2f}" for a in data) + f"   {gap:+.2f}")
    print("-- per draft position: acceptance rate (accepted at position i / drafts)")
    for a, rows in data.items():
        rs = [r for r in rows if r["mode"] == mode]
        d = sum(r["drafts"] for r in rs)
        by = collections.defaultdict(float)
        for r in rs:
            for k, v in r["pos_rate"].items():
                by[int(k)] += v * r["drafts"]
        print(f"   {a:11s} " + " ".join(f"p{k}:{by[k] / d:.2f}" for k in sorted(by)))
    for d in ("code", "math", "prose", "other"):
        sds = [[r["acc_len"] for r in data[a] if r["mode"] == mode and r["domain"] == d]
               for a in ("dflash2-k3", "mtp3") if a in data]
        print(f"   {d:6s} spread over seeds (acc len per request, DF2 k3 / MTP3): "
              + " / ".join(f"{st.pstdev(v):.2f}" if v else "-" for v in sds))
