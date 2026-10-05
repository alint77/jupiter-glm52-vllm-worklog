#!/usr/bin/env bash
# Whole-call A/B of tiered_decode_moe builds (VLLM_TIERED_DECODE_DEFINES
# variants), interleaved, on a held node:  ab.sh <tag> "<defines>" ...
# Writes ab-<tag>.jsonl lines {variant, round, hot, cold, us}.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
export TORCH_EXTENSIONS_DIR=${VLLM_CACHE_ROOT}/torch_extensions
D=agent_space/experiments/2026-10-05-decode-handoff
tag=$1; shift
CELLS=${CELLS:-"0,1 0,2 0,3 4,1 9,1 4,2 9,2 14,2 4,3 9,3 20,1 9,0 14,0"}
out=${D}/ab-${tag}.jsonl; : > ${out}
for round in 1 2; do
  for v in "$@"; do
    VLLM_TIERED_DECODE_DEFINES="${v}" numactl --cpunodebind=0 --membind=0 .venv/bin/python \
      agent_space/experiments/2026-09-27-glm53-mtp7-profile/bench_fmt.py --fmt int4 \
      --pool-hot 1000 --pool-cold 100 --grid ${CELLS} 2>/dev/null | grep '^{' \
      | sed "s/^{/{\"variant\": \"${v:-base}\", \"round\": ${round}, /" >> ${out}
  done
done
.venv/bin/python - "${out}" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
best = collections.defaultdict(lambda: 1e9)
for r in rows:
    k = (r["variant"], r["hot"], r["cold"]); best[k] = min(best[k], r["us"])
vs = list(dict.fromkeys(r["variant"] for r in rows))
cells = list(dict.fromkeys((r["hot"], r["cold"]) for r in rows))
print("hot,cold " + " ".join(f"{v:>16}" for v in vs))
for h, c in cells:
    b = best[(vs[0], h, c)]
    print(f"{h:3d},{c:<4d} " + " ".join(
        f"{best[(v, h, c)]:7.1f} ({best[(v, h, c)] - b:+5.1f})" for v in vs))
PY
