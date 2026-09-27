#!/usr/bin/env bash
# Pinned-clock timing of the integrated vLLM tiered decode path per (hot, cold)
# cell, over 4 GPUs:  fixedclock_vllm.sh <outdir> "h,c" ...
set -uo pipefail
out=$1; shift
E=$(cd "$(dirname "$0")" && pwd)
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
cells=("$@")
for gpu in 0 1 2 3; do
  (
    for i in "${!cells[@]}"; do
      (( i % 4 == gpu )) || continue
      IFS=, read -r h c <<< "${cells[$i]}"
      node=$(${PY} /e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py "${gpu}")
      "${E}/run_bound.sh" "${gpu}" ncu --clock-control base --profile-from-start off --csv \
        --metrics gpu__time_duration.sum "${PY}" "${E}/bench_vllm_decode.py" --hot "${h}" --cold "${c}" \
        --reps 5 --numa-node "${node}" > "${out}/vllm-${h}-${c}.csv" 2> "${out}/vllm-${h}-${c}.err"
    done
  ) &
done
wait
echo done
