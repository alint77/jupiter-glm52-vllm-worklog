#!/usr/bin/env bash
# Paired A/B of the tier-overlap threshold on the 16K PyTorch coding suite.
#
#   ./submit-overlap-ab.sh [pairs]
#
# Both arms run back to back in one allocation so a pair cannot be split across
# nodes, and the order alternates so drift within an allocation cannot favour
# one side. `old` pins the threshold to 16, the pre-2026-08-30 behaviour; `new`
# leaves it at the default, which is now max_num_batched_tokens.

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-tier-cost-surface"
arm="${here}/arm-overlap-ab.sh"
model="${TIERED_MODEL_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16}"
profile="${repo}/agent_space/profiles/glm53-w4a16-2496.json"
pairs="${1:-3}"

[[ -s "${arm}" ]] || { printf 'missing arm: %s\n' "${arm}" >&2; exit 1; }
[[ -d "${model}" ]] || { printf 'model not found: %s\n' "${model}" >&2; exit 1; }
mkdir -p "${here}/snapshots" "${here}/ab"
snap="${here}/snapshots/arm-overlap-ab-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for ((i = 1; i <= pairs; i++)); do
  if (( i % 2 == 1 )); then
    first="old-r${i}"; first_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=16"
    second="new-r${i}"; second_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS="
  else
    first="new-r${i}"; first_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS="
    second="old-r${i}"; second_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=16"
  fi
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="tier-overlap-ab-r${i}" \
    --output="${here}/slurm-overlap-ab-r${i}-%j.out" \
    --error="${here}/slurm-overlap-ab-r${i}-%j.err" \
    --wrap "TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab ${first_env} \
              bash ${snap} ${first} 4 ${profile} mtp3
            TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab ${second_env} \
              bash ${snap} ${second} 4 ${profile} mtp3"
done
