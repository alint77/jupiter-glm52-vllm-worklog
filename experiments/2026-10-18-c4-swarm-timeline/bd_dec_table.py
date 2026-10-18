"""Decode step breakdown by requests in flight (n = 1..4, M = 4n) at ~5K and
~50K context: ms/step per category, mean over the 4 ranks, from
an/bd-<arm><n>-w<window>-d<dev>.json (nsys_bd.py).

    bd_dec_table.py [arm prefix, default dec] [windows, default 1,2]
"""
import json
import sys
from pathlib import Path

A = Path(__file__).parent / "an"
arm = sys.argv[1] if len(sys.argv) > 1 else "dec"
wins = (sys.argv[2] if len(sys.argv) > 2 else "1,2").split(",")
cols = []
for w in wins:
    for n in range(1, 5):
        fs = [A / f"bd-{arm}{n}-w{w}-d{d}.json" for d in range(4)]
        if all(f.exists() for f in fs):
            cols.append((n, w, [json.loads(f.read_text()) for f in fs]))
mean = lambda v: sum(v) / len(v)
cats = list(cols[0][2][0]["ms"])


def trim(r):
    """Drop steps over 1.3x the median step (a window's admission / prefill
    steps); category means over the rest."""
    tot = sorted(sum(s) for s in r["per_step_ms"])
    keep = [s for s in r["per_step_ms"] if sum(s) <= 1.3 * tot[len(tot) // 2]]
    r["ms"] = {c: mean([s[i] for s in keep]) for i, c in enumerate(cats)}
    r["period_ms"] = mean([sum(s) for s in keep])
    r["dropped"] = len(r["per_step_ms"]) - len(keep)


for _, _, d in cols:
    for r in d:
        trim(r)
print(f"{'ms/step (mean of 4 ranks)':<52}" + "".join(f"{f'n{n} w{w}':>9}" for n, w, _ in cols))
print(f"{'M (verify tokens)':<52}" + "".join(f"{4 * n:>9}" for n, _, _ in cols))
for c in sorted(cats, key=lambda c: -mean([mean([r["ms"][c] for r in d]) for _, _, d in cols])):
    print(f"{c:<52}" + "".join(f"{mean([r['ms'][c] for r in d]):9.2f}" for _, _, d in cols))
print(f"{'step period':<52}" + "".join(f"{mean([r['period_ms'] for r in d]):9.2f}" for _, _, d in cols))
print(f"{'steps kept / dropped (rank 0)':<52}" + "".join(
    f"{len(d[0]['per_step_ms']) - d[0]['dropped']:>5}/{d[0]['dropped']:<3}" for _, _, d in cols))
