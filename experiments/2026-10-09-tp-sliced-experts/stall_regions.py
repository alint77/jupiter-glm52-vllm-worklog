"""Group ncu SASS stall samples of one kernel into regions: the top
instructions by samples with their neighbourhood, and a per-opcode summary.

    stall_regions.py sass.csv [--top 25]
"""
import collections
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]
ia, isrc, iall, inot = (hdr.index("Address"), hdr.index("Source"),
                        hdr.index("Warp Stall Sampling (All Samples)"),
                        hdr.index("Warp Stall Sampling (Not-issued Samples)"))
iexe = hdr.index("Instructions Executed")
ins = []
for r in rows[2:]:
    if len(r) <= iall:
        continue
    try:
        ins.append((r[ia], r[isrc].strip(), int(r[iall] or 0), int(r[inot] or 0), int(r[iexe] or 0)))
    except ValueError:
        pass
tot = sum(x[2] for x in ins)
print(f"{len(ins)} instructions, {tot} samples")
op = collections.Counter()
for _, s, a, _, _ in ins:
    op[s.split()[0] if not s.startswith("@") else s.split()[1]] += a
print("by opcode:", ", ".join(f"{k.split('.')[0]}:{v / tot:.1%}" for k, v in op.most_common(12)))
for i, (addr, s, a, n, e) in sorted(enumerate(ins), key=lambda x: -x[1][2])[:int(sys.argv[3]) if len(sys.argv) > 3 else 25]:
    print(f"{a / tot:6.1%} {addr} exe {e:7d}  {s[:90]}")
