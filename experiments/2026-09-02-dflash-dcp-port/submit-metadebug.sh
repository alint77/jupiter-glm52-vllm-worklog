#!/usr/bin/env bash
# Instrumented DCP4 arm: dump (batch, causal, num_splits) at the point
# scheduler_metadata is built and at the point it is consumed.
#
# FA3's metadata_size was measured empirically to depend on exactly three
# things -- batch, causal and num_splits -- and on nothing else (head counts,
# seqlens, headdim all leave it unchanged). So one of those three differs
# between flash_attn.py:601 and :1223. This arm says which.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-metadebug-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"
echo "frozen: ${snap}"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=01:00:00 \
  --job-name=df2-metadebug \
  --output="${here}/results/slurm-metadebug-%j.out" \
  --error="${here}/results/slurm-metadebug-%j.err" \
  --wrap "RESULT_DIR=${here}/results DCP=4 SEQS=4 VLLM_DCP_META_DEBUG=1 \
            VLLM_DFLASH2_PROBE=999999 \
            bash ${snap} metadebug dflash2 ${profile} gsm8k"
