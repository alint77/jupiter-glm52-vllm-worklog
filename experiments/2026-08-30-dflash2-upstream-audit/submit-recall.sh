#!/usr/bin/env bash
# Candidate-recall diagnostics: the measurement that bounds everything the
# selector can do. The unary-walk A/B cleared the lattice, so the position-0
# fault sits in the draft's hidden states / candidate computation -- or on
# the verify side. Two probes:
#
#   1. topk64: widen the candidate set to 64. If position-0 acceptance jumps,
#      the target's token was being recalled outside the checkpoint's top-16;
#      if it does not move, the draft's hidden states rank the true token low
#      regardless of width.
#   2. recall log: at verify time, check whether the target's argmax falls
#      inside the candidate set proposed one step earlier (VLLM_DFLASH2_RECALL_LOG).
#      The ground truth the topk64 probe only bounds. Run at the checkpoint's
#      top-16 on both GSM8K and HumanEval for the task contrast.
#
# Lattice control at top-16 on GSM8K already exists from the unary batch
# (lattice-ctrl, 3.9880, position 0 = 0.618).

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-recall-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-topk64-gsm8k \
  --output="${here}/results/slurm-topk64-gsm8k-%j.out" \
  --error="${here}/results/slurm-topk64-gsm8k-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_SELECTOR_TOPK=64 \
            bash ${snap} topk64-gsm8k dflash2 ${profile} gsm8k"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-recall-gsm8k \
  --output="${here}/results/slurm-recall-gsm8k-%j.out" \
  --error="${here}/results/slurm-recall-gsm8k-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_RECALL_LOG=1 \
            bash ${snap} recall-gsm8k dflash2 ${profile} gsm8k"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-recall-humaneval \
  --output="${here}/results/slurm-recall-humaneval-%j.out" \
  --error="${here}/results/slurm-recall-humaneval-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_RECALL_LOG=1 \
            bash ${snap} recall-humaneval dflash2 ${profile} humaneval"
