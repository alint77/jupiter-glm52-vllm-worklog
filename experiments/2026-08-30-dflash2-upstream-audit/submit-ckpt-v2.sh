#!/usr/bin/env bash
# Draft-checkpoint revision A/B: the model card's numbers describe a checkpoint
# we were not running.
#
# incoai/GLM-5.3-DFlash2 has two commits:
#   bae18bbff1 "Release"           2026-08-28 15:06Z  sha256 8ed9d14a...  <- ours
#   425aa615ce "Checkpoint update" 2026-08-28 21:12Z  sha256 3105f140...  <- current
#
# All 96 tensors differ between them (same config/shapes/keys, every weight
# retrained). The selector codebooks moved most (rel ~25-27%) -- precisely the
# stage that sets candidate quality, and precisely where our recall measurement
# found the fault (target token outside the top-16 36% of the time at position
# 0 on GSM8K).
#
# Two benchmarks per the standing rule, greedy, card protocol, so this is
# directly comparable to every prior arm.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-ckptv2-$(date +%Y%m%d-%H%M%S).sh"
# Point the arm at the updated draft checkpoint.
sed 's#/e/project1/profound/alint77/models/GLM-5.3-DFlash2#/e/project1/profound/alint77/models/GLM-5.3-DFlash2-v2#g' \
  "${arm}" > "${snap}"
chmod +x "${snap}"
grep -q "GLM-5.3-DFlash2-v2" "${snap}" || { echo "FATAL: snapshot not repointed"; exit 1; }

for task in gsm8k humaneval; do
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
    --job-name="df2-v2-${task}" \
    --output="${here}/results/slurm-v2-${task}-%j.out" \
    --error="${here}/results/slurm-v2-${task}-%j.err" \
    --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_RECALL_LOG=1 \
              bash ${snap} v2-${task} dflash2 ${profile} ${task}"
done
