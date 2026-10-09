#!/usr/bin/env bash
# Same-GPU variant comparison: ab.sh <tag> <m> "<cells>" "<variant>|<flags>" ...
# GPUs 0..3 each run every variant, GPUs 1 and 3 in reverse order; NUMA-bound.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
export ROOF_HBM=3600 ROOF_C2C=419
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
tag=$1 m=$2 cells=$3; shift 3
specs=("$@")
vs=(); for sp in "${specs[@]}"; do vs+=("${sp%%|*}"); done
# v13+ (no torch headers) build in parallel in seconds; older ones serially
$PY -c "import sys, kdev; kdev.build_many(sys.argv[1:]); [kdev.build(v) for v in sys.argv[1:]]" "${vs[@]}" >/dev/null 2>&1
rev=(); for ((i=${#specs[@]}-1; i>=0; i--)); do rev+=("${specs[i]}"); done
for g in 0 1 2 3; do
  if (( g % 2 )); then order=("${rev[@]}"); else order=("${specs[@]}"); fi
  ( for sp in "${order[@]}"; do v=${sp%%|*}; fl=${sp#*|}; [[ $fl == "$sp" ]] && fl=""
      CUDA_VISIBLE_DEVICES=$g numactl --cpunodebind=$g --membind=$g $PY kdev.py bench --v "$v" --m $m \
        --numa-node $g --cells $cells $fl 2>>logs/ab-$tag.err | grep "^{" | sed "s/}$/, \"gpu\": $g}/"
    done ) >> logs/ab-$tag.jsonl &
done
wait
$PY ab_sum.py logs/ab-$tag.jsonl; exit 0
: <<'PY'
import json, sys, collections, statistics as st
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); d[(r["v"].split(":")[0] + ("+side" if r["shared"] == "side" else ""), r["hot"], r["cold"])].append(r)
cells = sorted({(k[1], k[2]) for k in d})
vs = sorted({k[0] for k in d})
print("variant".ljust(14) + "".join(f"{h:>4}/{c:<3}  us  roof".rjust(18) for h, c in cells))
for v in vs:
    line = v.ljust(14)
    for h, c in cells:
        rs = d.get((v, h, c), [])
        if rs:
            line += f"{st.median(r['us'] for r in rs):10.1f} {st.median(r.get('roof_share', 0) for r in rs):6.2f}  "
        else:
            line += " " * 18
    print(line)
PY
