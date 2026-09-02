#!/usr/bin/env bash
# The 2x2 that separates DCP from concurrency, run at 32K instead of 400K.
#
# At the production 400K shape the two are inseparable: a fork guard makes
# max_num_seqs > 1 unreachable without DCP, because the replicated MLA cache
# does not fit two 400K sequences per rank. That guard is memory arithmetic,
# not physics, so shrink the context and all four cells fit.
# VLLM_TIERED_MOE_RELAX_SHAPE lifts the shape pins for diagnostics only.
#
# GSM8K prompts are a few hundred tokens with max_tokens 4096, so 32K is
# ample and nothing is truncated.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-2x2b-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 \
              VLLM_TIERED_MOE_RELAX_SHAPE=1 VLLM_DFLASH2_PROBE=999999 \
              EXTRA_ARGS='--max-model-len 32768' \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}
submit g-dcp1-c1 1 1
submit g-dcp1-c4 1 4
submit g-dcp4-c1 4 1
submit g-dcp4-c4 4 4
