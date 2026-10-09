"""Innermost MMA loops of a kernel's SASS: instructions per HMMA and the mix.

    sass_loops.py kdev/<name>/sass.txt [function-substring]
"""
import collections
import re
import sys

txt = open(sys.argv[1]).read()
fn = sys.argv[2] if len(sys.argv) > 2 else "layer_kernel"
start = txt.index(next(l for l in txt.splitlines() if "Function :" in l and fn in l))
end = txt.find("Function :", start + 10)
f = txt[start:end if end > 0 else None]
ops = []
for l in f.splitlines():
    m = re.match(r"\s+/\*([0-9a-f]{4,5})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.]+)", l)
    if m:
        ops.append((int(m.group(1), 16), m.group(3), l))
print(f"{fn}: {len(ops)} instructions")
loops = []
for addr, o, l in ops:
    if o.startswith("BRA"):
        m = re.search(r"0x([0-9a-f]+)", l.split("BRA", 1)[1])
        if m and int(m.group(1), 16) < addr:
            loops.append((int(m.group(1), 16), addr))
seen = []
for lo, hi in sorted(loops, key=lambda x: x[1] - x[0]):
    body = [o for a, o, _ in ops if lo <= a <= hi]
    nh = sum(o.startswith("HMMA") for o in body)
    if not nh or any(lo <= s0 and hi >= s1 for s0, s1 in seen):
        continue
    seen.append((lo, hi))
    c = collections.Counter(o.split(".")[0] for o in body)
    kinds = collections.Counter(o for o in body if o.startswith("HMMA"))
    print(f"loop {lo:05x}-{hi:05x}: {len(body)} instr, {dict(kinds)}, {len(body) / nh:.1f} per HMMA")
    print("   " + ", ".join(f"{k}:{v}" for k, v in c.most_common(14)))

# straight-line MMA blocks: windows from the first to last HMMA of each run of
# same-kind HMMAs no further than 80 instructions apart
idx = [i for i, (_, o, _) in enumerate(ops) if o.startswith("HMMA")]
runs, cur = [], [idx[0]] if idx else []
for i in idx[1:]:
    if i - cur[-1] <= 80 and ops[i][1] == ops[cur[-1]][1]:
        cur.append(i)
    else:
        runs.append(cur)
        cur = [i]
if cur:
    runs.append(cur)
for r in runs:
    if len(r) < 8:
        continue
    span = ops[r[0]:r[-1] + 1]
    c = collections.Counter(o.split(".")[0] for _, o, _ in span)
    print(f"block {ops[r[0]][0]:05x}-{ops[r[-1]][0]:05x}: {len(r)} x {ops[r[0]][1]}, "
          f"{len(span)} instr, {len(span) / len(r):.1f} per HMMA")
    print("   " + ", ".join(f"{k}:{v}" for k, v in c.most_common(12)))
