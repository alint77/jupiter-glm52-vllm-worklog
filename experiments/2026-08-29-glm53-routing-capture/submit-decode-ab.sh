#!/usr/bin/env bash
# Decode-isolating A/B: 256-token prompts, 1024-token generations, so prefill
# chunking cannot contaminate TPOT the way it does on the 16K suite.
set -euo pipefail
label="${1:?label}"; profile="${2:?profile}"; concurrency="${3:-4}"; mode="${4:-mtp3}"
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture"
[[ -s "${profile}" ]] || { printf 'profile not found: %s\n' "${profile}" >&2; exit 1; }
mkdir -p "${here}/snapshots"
snap="${here}/snapshots/arm-decode-${label}-$(date +%H%M%S).sh"
cp "${here}/${ARM:-arm-decode-only.sh}" "${snap}"
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 --gres=gpu:4 \
  --cpus-per-task=288 --time=02:00:00 --job-name="${label}" \
  --output="${here}/slurm-${label}-%j.out" --error="${here}/slurm-${label}-%j.err" \
  --wrap "srun --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --mem=0 \
            bash ${snap} ${label} ${concurrency} ${profile} ${mode}"
