#!/usr/bin/env bash
# Does buying HBM residency pay? Open since Phase 32, and never actually tested:
# a profile's slot count does not bind residency unless
# VLLM_TIERED_MOE_PROFILE_CAP=1, which nothing sets, so every earlier attempt
# compared rankings at identical residency.
#
# Four arms on ONE allocation, so the curve is free of node-to-node variation:
#
#   cap1800 / cap2100 / cap2400   PROFILE_CAP=1, residency == the profile count
#   uncapped                      PROFILE_CAP=0, residency == whatever HBM allows
#
# The three capped profiles are built from the same traces at different
# --hot-slots-per-rank, and their hot sets are verified nested, so the ranking is
# held fixed and only the count moves. The uncapped arm reuses cap-2400 and lets
# _promote_underfilled_residency fill up to the HBM limit, which is what
# production does today.

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-glm53-cc-capture"
arm="${repo}/agent_space/experiments/2026-08-29-glm53-w4a16/arm-quant-ab-2496.sh"
model="${TIERED_MODEL_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16}"

mkdir -p "${here}/snapshots" "${here}/residency/ab"
snap="${here}/snapshots/arm-residency-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

cmd=""
for n in 1800 2100 2400; do
  cmd+="TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/residency/ab \
          VLLM_TIERED_MOE_PROFILE_CAP=1 \
          bash ${snap} cap${n} 4 ${here}/residency/cap-${n}.json mtp3
"
done
cmd+="TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/residency/ab \
        VLLM_TIERED_MOE_PROFILE_CAP=0 \
        bash ${snap} uncapped 4 ${here}/residency/cap-2400.json mtp3
"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
  --job-name=glm53-residency \
  --output="${here}/residency/slurm-%j.out" \
  --error="${here}/residency/slurm-%j.err" \
  --wrap "${cmd}"
