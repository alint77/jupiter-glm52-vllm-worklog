#!/usr/bin/env bash
# Fixed-clock comparison (ncu --clock-control base pins SMs at ~1.41 GHz, the
# clock this GPU sustains under full MoE load): production Marlin per tier and
# count, and the tiered kernel per (hot, cold) cell. Spreads jobs over 4 GPUs.
#   fixedclock_sweep.sh <outdir> <sk binary>
set -uo pipefail
out=$1; skbin=$2
E=$(cd "$(dirname "$0")" && pwd)
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
mkdir -p "${out}"
NCU=(ncu --clock-control base --csv --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second)
jobs_for_gpu() {
  local gpu=$1; shift
  for j in "$@"; do
    set -- ${j//,/ }
    if [[ $1 == marlin ]]; then
      "${E}/run_bound.sh" "${gpu}" "${NCU[@]}" --profile-from-start off "${PY}" "${E}/marlin_mimo.py" \
        --tier "$2" --count "$3" --reps 5 --numa-node "$(${PY} /e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py ${gpu})" \
        > "${out}/marlin-$2-$3.csv" 2> "${out}/marlin-$2-$3.err"
    else
      "${E}/run_bound.sh" "${gpu}" "${NCU[@]}" -k regex:sk_kernel -c 8 "${E}/${skbin}" bench "$2" "$3" "$4" "$5" 4 0 \
        > "${out}/sk-$2-$3-$4-$5.csv" 2> "${out}/sk-$2-$3-$4-$5.err"
    fi
  done
}
all=()
for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16; do all+=("marlin,hot,${n}"); done
for n in 1 2 3 4 5; do all+=("marlin,cold,${n}"); done
for g in w13 w2; do
  for h in 0 3 5 7 9 11 13 16; do all+=("sk,${g},${h},0,0"); done
  for h in 0 5 7 9 11 13; do for c in 1 2 3 4; do
    cc=$(( c == 1 ? 16 : 24 )); all+=("sk,${g},${h},${c},${cc}"); done; done
done
for gpu in 0 1 2 3; do
  mine=(); for i in "${!all[@]}"; do (( i % 4 == gpu )) && mine+=("${all[$i]}"); done
  jobs_for_gpu "${gpu}" "${mine[@]}" &
done
wait
echo done
