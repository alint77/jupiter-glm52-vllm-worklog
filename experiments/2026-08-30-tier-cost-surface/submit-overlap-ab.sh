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
# The shipped profile. Its slot count is NOT what sets residency: the planner
# computes hot_slots = min(all owned, available_hbm / expert_bytes) and then
# pads or trims the profile's list to match (_promote_underfilled_residency /
# _demote_overfilled_residency). VLLM_TIERED_MOE_PROFILE_CAP would make the
# count binding, and nothing sets it -- not the arms, not prod. So swapping
# 2496 for 2400 frees no HBM, which is why the previous attempt failed
# identically.
profile="${PROFILE:-${repo}/agent_space/profiles/glm53-w4a16-2496.json}"
pairs="${1:-3}"

[[ -s "${arm}" ]] || { printf 'missing arm: %s\n' "${arm}" >&2; exit 1; }
[[ -d "${model}" ]] || { printf 'model not found: %s\n' "${model}" >&2; exit 1; }
mkdir -p "${here}/snapshots" "${here}/ab"
snap="${here}/snapshots/arm-overlap-ab-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for ((i = 1; i <= pairs; i++)); do
  if (( i % 2 == 1 )); then
    first="old${NEW_CAP:-8192}-r${i}"; first_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=16"
    second="new${NEW_CAP:-8192}-r${i}"; second_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=${NEW_CAP:-8192}"
  else
    first="new${NEW_CAP:-8192}-r${i}"; first_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=${NEW_CAP:-8192}"
    second="old${NEW_CAP:-8192}-r${i}"; second_env="VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=16"
  fi
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="tier-ab-${NEW_CAP:-8192}-r${i}" \
    --output="${here}/slurm-ab-${NEW_CAP:-8192}-r${i}-%j.out" \
    --error="${here}/slurm-ab-${NEW_CAP:-8192}-r${i}-%j.err" \
    --wrap "TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab TIERED_MOE_HBM_RESERVE_GB=${RESERVE_GB:-9} ${first_env} \
              bash ${snap} ${first} 4 ${profile} mtp3
            TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab TIERED_MOE_HBM_RESERVE_GB=${RESERVE_GB:-9} ${second_env} \
              bash ${snap} ${second} 4 ${profile} mtp3"
done
