#!/usr/bin/env bash
# GLM-5.3 W4A16 + MTP K=7, c=1, 400K context. Defaults are the settled
# baseline: DCP4, exact replicas, the INT4 one-kernel decode MoE, reserve 7,
# capture [1, 8] (REPLICAS= / DCP=1 / VLLM_TIERED_MOE_DECODE_KERNEL=0 undo them).
# Started from the spec-comparison
# mtp7 arm (../2026-09-04-spec-comparison/arm.sh) plus the production
# launcher's cold prefetch, with
#   * prefix caching off, so a repeated prompt really prefills
#   * --profiler-config when TRACE_ROOT is set
# MTP verifies 8 tokens per step, the same shape as MiMo's DFlash K=7.
# Run on the node from the repo root with the environment loaded.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
c=/e/fscratch/profound/${USER}/caches
export VLLM_CACHE_ROOT=${c}/marlin/vllm-cache-glm53-mtp7
export TRTLLM_DG_CACHE_DIR=${c}/marlin/trtllm-dg-glm53-mtp7
export TRITON_CACHE_DIR=${c}/triton
export TORCHINDUCTOR_CACHE_DIR=${c}/inductor
export FLASHINFER_CACHE_DIR=${c}/flashinfer
export TIERED_MOE_MODEL_PATH=/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16
export TIERED_MOE_PLACEMENT_PROFILE=${PWD}/agent_space/profiles/${PROFILE:-glm53-w4a16-2496.json}
export TIERED_MOE_HBM_RESERVE_GB="${RESERVE_GB:-7}"  # 10 was for DFlash2's draft KV
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024
# one-kernel INT4 decode MoE, replicas balanced by time in the kernel
export VLLM_TIERED_MOE_DECODE_KERNEL="${VLLM_TIERED_MOE_DECODE_KERNEL:-1}"
# [1, 8]: 8 is the verify step; 1 is MTP's draft-decode passes (positions
# 1..K-1, one token each at c=1). Without a size-1 entry the speculator's
# decode graph is silently skipped and those passes run eagerly.
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES:-1,8}],\"compile_sizes\":[8],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
extra=()
# REPLICAS=exact activates the profile's Grace replicas (985 per rank in
# glm53-w4a16-2496.json). Replicas add pinned Grace the planner does not see
# each worker's own share of, hence the larger host reserve (as for MiMo).
REPLICAS="${REPLICAS-exact}"
if [[ -n "${REPLICAS}" ]]; then
  extra+=(--tiered-moe-replica-assignment "${REPLICAS}"
          --tiered-moe-host-reserve-gb "${HOST_RESERVE_GB:-16}")
fi
if [[ -n "${TRACE_ROOT:-}" ]]; then
  mkdir -p "${TRACE_ROOT}"
  extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TRACE_ROOT}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}")
fi
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}' \
  --decode-context-parallel-size "${DCP:-4}" \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 400000 \
  --no-enable-prefix-caching \
  --generation-config vllm \
  --served-model-name glm53-w4a16-tiered \
  "${extra[@]}" ${SERVE_EXTRA:-}
