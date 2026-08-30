#!/usr/bin/env bash
# Final validation: do the fixes, together, make DFlash2 work properly?
#
# Two arms per benchmark on one allocation, all fixes present in both:
#   fixed-greedy  draft_sample_method=greedy        (matches upstream vLLM)
#   fixed-prob    draft_sample_method=probabilistic (matches SGLang, the 5.94 stack)
#
# Three benchmarks because the gap is already known to be non-uniform --
# HumanEval sits 10% under the card, GSM8K 32% under, 16K code collapses to
# 2.84. A change that moves only one of them has not fixed anything, and
# judging on GSM8K alone would judge on the least representative shape.
#
# Reference points: Phase 43 greedy 4.0293 (GSM8K), our MTP7 control 4.9121 at
# the same width, card 5.94 GSM8K / 5.48 HumanEval.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-validation-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for task in gsm8k humaneval longcode; do
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="dflash2-val-${task}" \
    --output="${here}/results/slurm-val-${task}-%j.out" \
    --error="${here}/results/slurm-val-${task}-%j.err" \
    --wrap "RESULT_DIR=${here}/results DRAFT_SAMPLE_METHOD=greedy \
              bash ${snap} fixedgreedy-${task} dflash2 ${profile} ${task}
            RESULT_DIR=${here}/results DRAFT_SAMPLE_METHOD=probabilistic \
              bash ${snap} fixedprob-${task} dflash2 ${profile} ${task}"
done
