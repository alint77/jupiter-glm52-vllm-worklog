#!/usr/bin/env bash
# Submit the four comparison arms as four parallel full-node jobs.
# Names stay within squeue's 8-character name column.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${here}"

submit() {  # name mode K
  local name="$1" mode="$2" width="$3"
  sbatch --parsable --account=profound --partition=booster \
    --nodes=1 --ntasks=1 --gres=gpu:4 --time=03:00:00 \
    --job-name="${name}" \
    --output="${here}/slurm-${name}-%j.out" \
    --error="${here}/slurm-${name}-%j.err" \
    --wrap "RESULT_DIR=${here} ${here}/arm.sh ${name} ${mode} ${width}"
}

submit spdf23 dflash 3
submit spdf27 dflash 7
submit spmtp3 mtp 3
submit spmtp7 mtp 7
