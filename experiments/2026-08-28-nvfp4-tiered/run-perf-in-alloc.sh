#!/usr/bin/env bash
set -euo pipefail
alloc="${1:?alloc}"; label="${2:?label}"; conc="${3:-4}"; profile="${4:?profile}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"
git diff --quiet || { echo "REFUSING: tracked files modified"; exit 1; }
state="$(squeue -j "${alloc}" -h -o %T 2>/dev/null || true)"
[[ "${state}" == RUNNING ]] || { echo "alloc ${alloc} not RUNNING (${state:-gone})"; exit 1; }
cpus="$(scontrol show job "${alloc}" | tr ' ' '\n' | sed -n 's/^NumCPUs=//p' | head -1)"
echo "running ${label} c=${conc} in alloc ${alloc} ($(git rev-parse --short HEAD))"
exec srun --jobid="${alloc}" --overlap --nodes=1 --ntasks=1 --mpi=none \
  --cpus-per-task="${cpus}" --gres=gpu:4 --mem=0 \
  --export=ALL,TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-}" \
  bash "${result_dir}/arm-nvfp4-perf.sh" "${label}" "${conc}" "${profile}"
