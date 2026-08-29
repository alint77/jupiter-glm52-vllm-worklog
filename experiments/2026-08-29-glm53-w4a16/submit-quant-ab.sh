#!/usr/bin/env bash
# A/B two GLM-5.3 quantizers on the real-code decode suite, holding the
# placement ranking, concurrency, speculator and prompts fixed so the only
# variable is the checkpoint.
#
#   ./submit-quant-ab.sh <label> <model-dir> <profile> [concurrency] [mode]
set -euo pipefail
label="${1:?label}"; model="${2:?model dir}"; profile="${3:?profile}"
concurrency="${4:-4}"; mode="${5:-mtp3}"
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-29-glm53-w4a16"
[[ -d "${model}" ]] || { printf 'model not found: %s\n' "${model}" >&2; exit 1; }
[[ -s "${profile}" ]] || { printf 'profile not found: %s\n' "${profile}" >&2; exit 1; }
mkdir -p "${here}/snapshots"
snap="${here}/snapshots/arm-${label}-$(date +%H%M%S).sh"
cp "${here}/${ARM:-arm-quant-ab.sh}" "${snap}"
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=288 --time=02:00:00 --job-name="${label}" \
  --output="${here}/slurm-${label}-%j.out" --error="${here}/slurm-${label}-%j.err" \
  --wrap "TIERED_MODEL_DIR=${model} srun --nodes=1 --ntasks=1 --cpus-per-task=288 \
            --gres=gpu:4 --mem=0 bash ${snap} ${label} ${concurrency} ${profile} ${mode}"
