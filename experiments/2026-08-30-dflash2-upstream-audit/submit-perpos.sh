#!/usr/bin/env bash
# Per-position acceptance decomposition, the diagnostic the user asked for.
#
# Acceptance length alone cannot localise a draft-accuracy fault. The
# per-position survival curve can:
#
#   position 0 already low   -> the draft's INPUTS are wrong (aux states,
#                               context KV, embeddings), because position 0 is
#                               the same next-token task MTP performs well at;
#   position 0 ~ MTP, deep collapse -> the LATTICE / path walk is the problem;
#   flat geometric decay     -> nothing structurally wrong, the drafter is
#                               simply weaker than the card's.
#
# Run for DFlash2 (both sampling modes) and MTP7 as the control, one allocation
# each, so the curves are directly comparable at the same width.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-perpos-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

# DFlash2 greedy + probabilistic
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-perpos-greedy \
  --output="${here}/results/slurm-perpos-greedy-%j.out" \
  --error="${here}/results/slurm-perpos-greedy-%j.err" \
  --wrap "RESULT_DIR=${here}/results DRAFT_SAMPLE_METHOD=greedy \
            bash ${snap} perpos-greedy dflash2 ${profile} gsm8k"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-perpos-prob \
  --output="${here}/results/slurm-perpos-prob-%j.out" \
  --error="${here}/results/slurm-perpos-prob-%j.err" \
  --wrap "RESULT_DIR=${here}/results DRAFT_SAMPLE_METHOD=probabilistic \
            bash ${snap} perpos-prob dflash2 ${profile} gsm8k"

# MTP7 control: same width, same protocol
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-perpos-mtp \
  --output="${here}/results/slurm-perpos-mtp-%j.out" \
  --error="${here}/results/slurm-perpos-mtp-%j.err" \
  --wrap "RESULT_DIR=${here}/results \
            bash ${snap} perpos-mtp7 mtp3 ${profile} gsm8k"
