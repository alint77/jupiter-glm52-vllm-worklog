#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/jupiter-env.sh"

exec "${VLLM_REPO_DIR}/.venv/bin/vllm" serve "${GLM52_W4A16_MODEL}" \
  --served-model-name glm52-w4a16 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --distributed-executor-backend mp \
  --numa-bind \
  --offload-backend uva \
  --cpu-offload-gb 40 \
  --cpu-offload-params experts \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len 400000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 2 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[1],"pass_config":{"fuse_allreduce_rms":false}}'
