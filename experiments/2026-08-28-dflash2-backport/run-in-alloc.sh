#!/usr/bin/env bash
# Run one DFlash2 arm inside a standing salloc allocation, skipping the queue.
#
#   salloc --account=profound --partition=booster --nodes=1 --ntasks=1 \
#          --gres=gpu:4 --time=04:00:00 --no-shell
#   ./run-in-alloc.sh <alloc-job-id> [label] [spec_tokens]
#
# Bring-up is a tight edit/run loop and queue time dominates it, so the
# allocation is held and each attempt is just a step inside it. git does not
# exist on Booster compute nodes, so the tree is checked here, on the login
# node, exactly as submit.sh does for the batch path.
#
# --mpi=none: this launches one plain process, not an MPI job. The engine uses
# the `mp` executor internally and never calls srun, and after the psid/psslurm
# migration a PMIx bootstrap here would only be a way to break that.
set -euo pipefail

alloc="${1:?allocation job id (from salloc --no-shell)}"
label="${2:-dflash2}"
spec_tokens="${3:-7}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-dflash2-backport"
cd "${repo_dir}"

git diff --quiet || { echo "REFUSING: tracked files are modified"; git status --short; exit 1; }
git diff --cached --quiet || { echo "REFUSING: staged changes present"; exit 1; }

state="$(squeue -j "${alloc}" -h -o %T 2>/dev/null || true)"
[[ "${state}" == RUNNING ]] || { echo "allocation ${alloc} is not RUNNING (state: ${state:-gone})"; exit 1; }

echo "running ${label} as a step in allocation ${alloc} ($(git rev-parse --short HEAD))"
exec srun --jobid="${alloc}" --overlap --nodes=1 --ntasks=1 --mpi=none \
  bash "${result_dir}/arm-dflash2.sh" "${label}" "${spec_tokens}"
