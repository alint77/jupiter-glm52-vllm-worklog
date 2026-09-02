#!/usr/bin/env bash
# Is the DCP4 scheduler_metadata failure FlashAttention-specific?
#
# Phase 53's prod arms died at engine init in flash_attn.py:1223
# _forward_with_dcp. The draft reaching that path at all is the #52188 port
# working; the question is whether FA's DCP context path simply rejects the
# draft's max_query_len == 8 shape. flashinfer and triton both drove the
# draft successfully in Phase 52 (at DCP1), so they isolate the backend.
#
# A DCP1 control on the same backend runs alongside each, so a failure that
# has nothing to do with DCP is not misread as one that does.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-bisect-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"
echo "frozen: ${snap}"

submit() {  # name dcp seqs extra_env
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 \
    --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 $4 \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}

submit fi-dcp4 4 4 "DRAFT_ATTN=FLASHINFER VLLM_DFLASH2_PROBE=999999"
submit fi-dcp1 1 1 "DRAFT_ATTN=FLASHINFER VLLM_DFLASH2_PROBE=999999"
submit tri-dcp4 4 4 "DRAFT_ATTN=TRITON_ATTN VLLM_DFLASH2_PROBE=999999"
submit tri-dcp1 1 1 "DRAFT_ATTN=TRITON_ATTN VLLM_DFLASH2_PROBE=999999"
