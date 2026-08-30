#!/usr/bin/env bash
# Paired A/B of the real-usage ranking against the shipped synthetic one.
#
#   ./submit-profile-ab.sh [pairs]
#
# Everything is held fixed except the placement profile: same W4A16 checkpoint,
# same 2496 hot slots per rank, same 3940 replicas, same c4/DCP4/MTP3, same
# real-code decode suite. Each job runs both arms back to back on one
# allocation, so a pair is immune to node-to-node variation; between-run spread
# on this configuration is about 2.7%, which is why three pairs and not one.

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-glm53-cc-capture"
arm="${repo}/agent_space/experiments/2026-08-29-glm53-w4a16/arm-quant-ab-2496.sh"
model="${TIERED_MODEL_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16}"
shipped="${repo}/agent_space/profiles/glm53-w4a16-2496.json"
candidate="${CANDIDATE_PROFILE:-${here}/results-snap-1535650-a/replicas-985.json}"
pairs="${1:-3}"

for p in "${shipped}" "${candidate}" "${arm}"; do
  [[ -s "${p}" ]] || { printf 'missing: %s\n' "${p}" >&2; exit 1; }
done
[[ -d "${model}" ]] || { printf 'model not found: %s\n' "${model}" >&2; exit 1; }

mkdir -p "${here}/snapshots" "${here}/ab"
snap="${here}/snapshots/arm-profile-ab-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for ((i = 1; i <= pairs; i++)); do
  # Alternate which arm runs first, so a warm-cache or thermal drift within an
  # allocation cannot favour one side systematically.
  if (( i % 2 == 1 )); then
    first_label="old-r${i}"; first="${shipped}"
    second_label="new-r${i}"; second="${candidate}"
  else
    first_label="new-r${i}"; first="${candidate}"
    second_label="old-r${i}"; second="${shipped}"
  fi
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="glm53-prof-ab-r${i}" \
    --output="${here}/slurm-prof-ab-r${i}-%j.out" \
    --error="${here}/slurm-prof-ab-r${i}-%j.err" \
    --wrap "TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab \
              srun --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --mem=0 \
                bash ${snap} ${first_label} 4 ${first} mtp3
            TIERED_MODEL_DIR=${model} RESULT_DIR=${here}/ab \
              srun --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --mem=0 \
                bash ${snap} ${second_label} 4 ${second} mtp3"
done
