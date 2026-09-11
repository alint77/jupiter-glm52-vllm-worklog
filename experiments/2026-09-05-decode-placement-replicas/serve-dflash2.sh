#!/usr/bin/env bash
set -euo pipefail
profile="$1"
assignment="$2"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="${PREFETCH_MIN_TOKENS:-0}"
export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=0
export VLLM_TIERED_MOE_ROUTE_CHECK="${ROUTE_CHECK:-0}"
export TIERED_MOE_MODEL_PATH="/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export TIERED_MOE_HBM_RESERVE_GB=10
if [[ -z "${TIERED_MOE_COMPILATION_CONFIG:-}" ]]; then
  TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
fi
export TIERED_MOE_COMPILATION_CONFIG
drafter="/e/fscratch/profound/${USER}/models/GLM-5.3-DFlash2"
exec bash agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}" \
  --decode-context-parallel-size 1 --max-num-seqs 1 \
  --tiered-moe-replica-assignment "${assignment}" \
  --served-model-name glm53-placement --port 8129 \
  --gpu-memory-utilization 0.90 --max-model-len 400000 \
  --generation-config vllm --no-enable-prefix-caching "${@:3}"
