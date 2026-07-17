#!/usr/bin/env bash

set -euo pipefail

source agent_space/jupiter-env.sh

placement_profile="${TIERED_MOE_PLACEMENT_PROFILE-agent_space/experiments/2026-07-17-trace-placement/tail-placement-profile.json}"
hbm_reserve_gb="${TIERED_MOE_HBM_RESERVE_GB:-7}"
placement_args=()
if [[ -n "${placement_profile}" ]]; then
  placement_args=(--tiered-moe-placement-profile "${placement_profile}")
fi

exec .venv/bin/vllm serve "${GLM52_W4A16_MODEL}" \
  --served-model-name glm52-w4a16-tiered \
  --host 127.0.0.1 \
  --port 8027 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --distributed-executor-backend mp \
  --numa-bind \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --max-model-len 400000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 2 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[1],"pass_config":{"fuse_allreduce_rms":false}}' \
  --enable-tiered-moe \
  --tiered-moe-backend uva \
  "${placement_args[@]}" \
  --tiered-moe-hbm-reserve-gb "${hbm_reserve_gb}" \
  --tiered-moe-host-reserve-gb 8 \
  --tiered-moe-numa-strict \
  --mla-cache-tier hbm \
  --grace-machine-profile agent_space/profiles/jupiter-gh200-baseline.json \
  "$@"
