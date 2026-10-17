"""Prepare one PTX entry for SentryWard.

Keeps the header and one .entry, drops `prefetch.tensormap` (a cache hint;
SentryWard 60a1ebc clobbers its operand register, losing the map pointer
the later TMA reuses), and with --assume-aligned rewrites the
`(p + 1023) & -1024` dynamic-smem realignment to `p + 0`, i.e. assumes the
dynamic shared base is already 1024-aligned (SentryWard cannot follow `and`
on a pointer). `bfi` with immediate position/length is lowered to
and/shl/or, which SentryWard's concrete interpreter evaluates.
"""
import re
import sys

src, entry, dst = sys.argv[1:4]
aligned = "--assume-aligned" in sys.argv
lines = open(src).read().split("\n")
starts = [i for i, l in enumerate(lines) if ".entry " in l]
first = starts[0]
k = next(i for i in starts if entry in lines[i])
end = next((i for i in starts if i > k), len(lines))
out = lines[:first] + lines[k:end]
out = [l for l in out if "prefetch.tensormap" not in l]
if aligned:
    pend = {}
    for i, l in enumerate(out):
        m = re.match(r"\s*add\.s64\s+(%rd\d+), (%rd\d+), 1023;", l)
        if m:
            pend[m.group(1)] = (i, m.group(2))
        m = re.match(r"\s*and\.b64\s+(%rd\d+), (%rd\d+), -1024;", l)
        if m and m.group(2) in pend:
            j, base = pend[m.group(2)]
            out[i] = f"\tadd.s64 \t{m.group(1)}, {base}, 0;"
bfi = re.compile(r"(\s*)bfi\.b32\s+(%r\d+), (%r\d+), (%r\d+), (\d+), (\d+);")
lowered = []
for l in out:
    m = bfi.match(l)
    if not m:
        lowered.append(l)
        continue
    ind, d, a, b, pos, n = m.groups()
    mask = ((1 << int(n)) - 1) << int(pos) & 0xFFFFFFFF
    lowered += [
        f"{ind}{{ .reg .b32 %bfa, %bfb;",
        f"{ind}shl.b32 %bfa, {a}, {pos};",
        f"{ind}and.b32 %bfa, %bfa, {mask};",
        f"{ind}and.b32 %bfb, {b}, {~mask & 0xFFFFFFFF};",
        f"{ind}or.b32 {d}, %bfa, %bfb; }}",
    ]
out = lowered
open(dst, "w").write("\n".join(out))
