#!/usr/bin/env bash
# A/B the re-derived GLM-5.3 ranking against the shipped GLM-5.2 placeholder on
# the same c4/MTP3/DCP4 suite that measured 159.73 output tok/s and 22.98 ms
# TPOT, so the payoff is read against a number that already exists.
#
#   ./submit-perf-ab.sh <label> <profile> [concurrency] [mode]
#
# The arm is snapshotted before submission: editing a live script under a
# running job cost three jobs to walltime once already.

set -euo pipefail

label="${1:?label}"
profile="${2:?placement profile}"
concurrency="${3:-4}"
mode="${4:-mtp3}"

repo=/e/project1/profound/alint77/vllm
rd="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered"
here="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture"

[[ -s "${profile}" ]] || { printf 'profile not found: %s\n' "${profile}" >&2; exit 1; }

mkdir -p "${here}/snapshots"
snap="${here}/snapshots/arm-perf-${label}-$(date +%H%M%S).sh"
cp "${rd}/arm-nvfp4-perf.sh" "${snap}"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=288 --time=02:00:00 --job-name="${label}" \
  --output="${here}/slurm-${label}-%j.out" --error="${here}/slurm-${label}-%j.err" \
  --wrap "srun --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --mem=0 \
            bash ${snap} ${label} ${concurrency} ${profile} ${mode}"
