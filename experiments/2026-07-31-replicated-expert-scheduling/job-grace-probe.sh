#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.err

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
experiment="${repo}/agent_space/experiments/2026-07-31-replicated-expert-scheduling"
output="${experiment}/grace-probe-${SLURM_JOB_ID}"

cd "${repo}"
source agent_space/jupiter-env.sh
mkdir -p "${output}"
read -r -a targets <<<"${GRACE_PROBE_TARGETS_GB:-55.209 77.870 98.746 100.531 111.581}"

pids=()
for gpu in 0 1 2 3; do
  numa="$(
    .venv/bin/python \
      agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py \
      "${gpu}"
  )"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    numactl --cpunodebind="${numa}" --membind="${numa}" \
    .venv/bin/python "${experiment}/grace_probe.py" \
      --device 0 \
      --numa-node "${numa}" \
      --targets-gb "${targets[@]}" \
      --output "${output}/rank-${gpu}.json" \
      >"${output}/rank-${gpu}.out" \
      2>"${output}/rank-${gpu}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

for file in "${output}"/rank-*.json; do
  cat "${file}"
done
exit "${status}"
