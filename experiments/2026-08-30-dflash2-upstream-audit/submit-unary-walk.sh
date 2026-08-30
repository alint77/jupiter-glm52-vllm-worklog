#!/usr/bin/env bash
# Unary-walk A/B: localise the position-0 acceptance fault.
#
# Per-position curves (2026-08-30, GSM8K card protocol) put DFlash2 at 0.615
# acceptance on position 0 against MTP7's 0.915 -- the same next-token task,
# same conditioning. The lattice score is unary + pairwise(anchor, h), so:
#
#   unary walk ~= lattice walk  -> the draft's hidden states / candidates are
#                                  the bottleneck (input problem);
#   unary walk >> lattice walk  -> the trained lattice is dragging the choice
#                                  off the unary argmax in our run (selector
#                                  problem: codebooks, projection, walk).
#
# VLLM_DFLASH2_SELECTOR_WALK=unary makes the selector take the top-1 candidate
# at every position (candidates are sorted), ignoring the codebook scores.
# Two benchmarks per the standing rule; one lattice control re-run on the
# current build so the comparison is same-build.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-unary-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

# Unary walk on two benchmarks
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-unary-gsm8k \
  --output="${here}/results/slurm-unary-gsm8k-%j.out" \
  --error="${here}/results/slurm-unary-gsm8k-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_SELECTOR_WALK=unary \
            bash ${snap} unary-gsm8k dflash2 ${profile} gsm8k"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-unary-humaneval \
  --output="${here}/results/slurm-unary-humaneval-%j.out" \
  --error="${here}/results/slurm-unary-humaneval-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_SELECTOR_WALK=unary \
            bash ${snap} unary-humaneval dflash2 ${profile} humaneval"

# Lattice control on the same build (greedy, GSM8K)
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=02:00:00 \
  --job-name=df2-lattice-ctrl \
  --output="${here}/results/slurm-lattice-ctrl-%j.out" \
  --error="${here}/results/slurm-lattice-ctrl-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_SELECTOR_WALK=lattice \
            bash ${snap} lattice-ctrl dflash2 ${profile} gsm8k"
