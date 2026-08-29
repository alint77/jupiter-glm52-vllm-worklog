#!/usr/bin/env bash
# GSM8K accuracy discrimination: is the r1 dip specific to the group-32 path,
# caused by prefix caching, or pre-existing warm-up behaviour?
#   ./submit-acc.sh <label> <model-dir> <profile> [disable_prefix_cache 0|1]
set -euo pipefail
label="${1:?label}"; model="${2:?model}"; profile="${3:?profile}"; nopc="${4:-0}"
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-29-glm53-w4a16"
mkdir -p "${here}/snapshots"
snap="${here}/snapshots/arm-acc-${label}-$(date +%H%M%S).sh"
cp "${here}/${ACC_ARM:-arm-gsm8k-3rep.sh}" "${snap}"
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=288 --time=03:00:00 --job-name="${label}" \
  --output="${here}/slurm-${label}-%j.out" --error="${here}/slurm-${label}-%j.err" \
  --wrap "DISABLE_PREFIX_CACHE=${nopc} srun --nodes=1 --ntasks=1 --cpus-per-task=288 \
            --gres=gpu:4 --mem=0 bash ${snap} ${label} ${MODE:-mtp3} ${profile} ${model}"
