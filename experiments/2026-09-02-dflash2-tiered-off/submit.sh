#!/usr/bin/env bash
# Three arms, one node each, submitted together: two offload sizes for
# DFlash2 (bracketing the HBM fit rather than bisecting it serially) and a
# matched MTP7 control on the same build.
set -euo pipefail
here=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-09-02-dflash2-tiered-off
mkdir -p "${here}/results"
submit() {
  local name="$1" mode="$2" off="$3"
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=03:00:00 \
    --job-name="${name}" \
    --output="${here}/results/slurm-${name}-%j.out" \
    --error="${here}/results/slurm-${name}-%j.err" \
    --wrap "bash ${here}/arm-tieredoff.sh ${name} ${mode} ${off} gsm8k"
}
submit tieredoff-dflash2-off40 dflash2 40
submit tieredoff-dflash2-off56 dflash2 56
submit tieredoff-mtp7-off40    mtp     40
