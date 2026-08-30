#!/usr/bin/env bash
# Does the null-block guard (#51538) fix DFlash2's draft accuracy?
#
# Three benchmarks, deliberately: the defect only fires when the draft's 2048
# sliding window evicts context, so GSM8K (short answers) should barely move,
# HumanEval a little, and 16K code the most. A single benchmark could show a
# gain that is noise or a regression that is shape-specific; the *ordering
# across context lengths* is the prediction being tested, and it is not
# something a benchmark can be tuned into.
#
# One allocation per benchmark, guard off then on, so each pair is same-node.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-guard-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for task in gsm8k humaneval longcode; do
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="dflash2-guard-${task}" \
    --output="${here}/results/slurm-guard-${task}-%j.out" \
    --error="${here}/results/slurm-guard-${task}-%j.err" \
    --wrap "RESULT_DIR=${here}/results VLLM_DFLASH_NULL_BLOCK_GUARD=0 \
              bash ${snap} guardoff-${task} dflash2 ${profile} ${task}
            RESULT_DIR=${here}/results VLLM_DFLASH_NULL_BLOCK_GUARD=1 \
              bash ${snap} guardon-${task} dflash2 ${profile} ${task}"
done
