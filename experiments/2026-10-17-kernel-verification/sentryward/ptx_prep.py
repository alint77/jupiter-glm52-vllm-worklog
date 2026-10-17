"""Prepare one PTX entry for SentryWard 60a1ebc, validating every rewrite.

Usage: ptx_prep.py <in.ptx> <entry-substring> <out.ptx> [--assume-aligned]
       [--expect-prefetch N] [--expect-align N] [--expect-bfi N]

Keeps the header and one .entry, then:
- drops `prefetch.tensormap [%rdN];` (a cache hint; SentryWard clobbers its
  operand register, losing the map pointer a later TMA reuses);
- with --assume-aligned, rewrites `(p + 1023) & -1024` to `p + 0`, only where
  p is the immediately preceding `cvta.shared` of a `mov` of a shared symbol
  declared `.align 1024` (SentryWard cannot follow `and` on a pointer);
- lowers `bfi.b32` with in-range immediate position/length to and/shl/or,
  which SentryWard's concrete interpreter evaluates.
Any site count differing from its --expect-* value is an error.
"""
import argparse
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("entry")
ap.add_argument("dst")
ap.add_argument("--assume-aligned", action="store_true")
for k in ("prefetch", "align", "bfi"):
    ap.add_argument(f"--expect-{k}", type=int)
a = ap.parse_args()


def fail(msg):
    sys.exit(f"ptx_prep: {msg}")


lines = open(a.src).read().split("\n")
starts = [i for i, l in enumerate(lines) if ".entry " in l]
if not starts:
    fail("no .entry")
picked = [i for i in starts if a.entry in lines[i]]
if len(picked) != 1:
    fail(f"{len(picked)} entries match {a.entry!r}")
k = picked[0]
end = next((i for i in starts if i > k), len(lines))
out = lines[: starts[0]] + lines[k:end]
counts = {"prefetch": 0, "align": 0, "bfi": 0}

kept = []
for l in out:
    if re.fullmatch(r"\s*prefetch\.tensormap \[%rd\d+\];", l):
        counts["prefetch"] += 1
    elif "prefetch.tensormap" in l:
        fail(f"unrecognized prefetch form: {l.strip()}")
    else:
        kept.append(l)
out = kept

if a.assume_aligned:
    aligned = {
        m.group(1)
        for l in out
        if (m := re.match(r"\.extern \.shared \.align 1024 \.b8 (\S+)\[\];", l))
    }
    for i, l in enumerate(out):
        m = re.fullmatch(r"\s*and\.b64\s+(%rd\d+), (%rd\d+), -1024;", l)
        if not m:
            continue
        add = re.fullmatch(r"\s*add\.s64\s+(%rd\d+), (%rd\d+), 1023;", out[i - 1])
        cvta = re.fullmatch(r"\s*cvta\.shared\.u64\s+(%rd\d+), %tmp; }", out[i - 2])
        cvt = re.fullmatch(r"\s*cvt\.u64\.u32\s+%tmp, (%r\d+);", out[i - 3])
        mov = re.fullmatch(r"\s*mov\.u32\s+(%r\d+), (\S+);", out[i - 5])
        if not (add and cvta and cvt and mov and add.group(1) == m.group(2)
                and cvta.group(1) == add.group(2) and cvt.group(1) == mov.group(1)
                and mov.group(2) in aligned):
            fail(f"realignment at PTX line {i + 1} is not the validated pattern")
        out[i] = f"\tadd.s64 \t{m.group(1)}, {add.group(2)}, 0;"
        counts["align"] += 1

bfi = re.compile(r"(\s*)bfi\.b32\s+(%r\d+), (%r\d+), (%r\d+), (\d+), (\d+);")
lowered = []
for l in out:
    m = bfi.fullmatch(l)
    if not m:
        if re.match(r"\s*bfi\.", l):
            fail(f"unsupported bfi form: {l.strip()}")
        lowered.append(l)
        continue
    ind, d, x, y, pos, n = m.groups()
    pos, n = int(pos), int(n)
    if not (0 <= pos < 32 and 1 <= n and pos + n <= 32):
        fail(f"bfi position/length out of range: {l.strip()}")
    mask = ((1 << n) - 1) << pos
    lowered += [
        f"{ind}{{ .reg .b32 %bfa, %bfb;",
        f"{ind}shl.b32 %bfa, {x}, {pos};",
        f"{ind}and.b32 %bfa, %bfa, {mask};",
        f"{ind}and.b32 %bfb, {y}, {~mask & 0xFFFFFFFF};",
        f"{ind}or.b32 {d}, %bfa, %bfb; }}",
    ]
    counts["bfi"] += 1
out = lowered

for key, n in counts.items():
    want = getattr(a, f"expect_{key}")
    if want is not None and n != want:
        fail(f"{key}: {n} sites, expected {want}")
open(a.dst, "w").write("\n".join(out))
print(f"ptx_prep: {counts}")
