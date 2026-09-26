#!/usr/bin/env bash
# Run a command on the standby Booster allocation (hold.sbatch), from the repo
# root with the environment loaded:  ./onnode.sh <cmd...>
# HOLD_JOB overrides the job id; otherwise the newest running dak-hold job.
set -euo pipefail
job="${HOLD_JOB:-$(squeue -h -u "${USER}" -n dak-hold -t RUNNING -o %i | sort -n | tail -1)}"
[[ -n "${job}" ]] || { echo "no running dak-hold job" >&2; exit 2; }
exec srun --jobid="${job}" --overlap --ntasks=1 --gres=gpu:4 bash -c \
  "cd /e/project1/profound/alint77/vllm && source agent_space/jupiter-env.sh >/dev/null 2>&1 && $*"
