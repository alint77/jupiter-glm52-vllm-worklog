#!/usr/bin/env bash
# Run the NVFP4 bring-up as a step in a standing allocation. See the sibling
# phase's runner for why: git does not exist on Booster compute nodes, so the
# tree is checked here, and a step gets one CPU by default unless told
# otherwise, which breaks vLLM's GPU-to-NUMA detection outright.
set -euo pipefail
alloc="${1:?allocation job id}"
label="${2:-nvfp4-a1}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"
git diff --quiet || { echo "REFUSING: tracked files are modified"; git status --short; exit 1; }
git diff --cached --quiet || { echo "REFUSING: staged changes present"; exit 1; }
state="$(squeue -j "${alloc}" -h -o %T 2>/dev/null || true)"
[[ "${state}" == RUNNING ]] || { echo "allocation ${alloc} is not RUNNING (${state:-gone})"; exit 1; }
cpus="$(scontrol show job "${alloc}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^NumCPUs=//p' | head -1)"
[[ -n "${cpus}" ]] || { echo "could not read NumCPUs"; exit 1; }
echo "running ${label} in allocation ${alloc} ($(git rev-parse --short HEAD)), ${cpus} CPUs"
exec srun --jobid="${alloc}" --overlap --nodes=1 --ntasks=1 --mpi=none \
  --cpus-per-task="${cpus}" --gres=gpu:4 --mem=0 \
  --export=ALL,TIERED_MOE_PLACEMENT_PROFILE="${TIERED_MOE_PLACEMENT_PROFILE:-}",TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-}" \
  bash "${result_dir}/arm-nvfp4.sh" "${label}"
