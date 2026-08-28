#!/usr/bin/env bash
# NVFP4 WITHOUT the tiered path, to establish the upstream baseline: does this
# checkpoint serve on GH200 through the stock Marlin NVFP4 backend, untouched?
# Separates "the fork's loader needs teaching" from "NVFP4 does not work here".
#
# kv-cache-dtype must be pinned: the checkpoint's hf_quant_config asks for
# FP8 KV, and vLLM then selects fp8_e4m3 for a head_size-576 MLA cache, for
# which no attention backend exists ("No valid attention backend found").
# fp8_ds_mla is the sparse-MLA layout this model actually needs.
#
# The model does not fit in 4x95 GiB, so this is expected to OOM on weights.
# What it proves is where it gets to first: reaching a memory limit means the
# format path is sound and only residency stands in the way.
set -euo pipefail
label="${1:?label}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"; source agent_space/jupiter-env.sh
echo "node: $(hostname)"; echo "label: ${label} (DENSE, no tiered MoE)"
model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-nvfp4-dense"
mkdir -p "${VLLM_CACHE_ROOT}"
.venv/bin/vllm serve "${model}" --served-model-name glm53-nvfp4 \
  --host 127.0.0.1 --port 8031 --tensor-parallel-size 4 --enable-expert-parallel \
  --distributed-executor-backend mp --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 --block-size 64 \
  --kv-cache-dtype fp8_ds_mla \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!
for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:8031/health >/dev/null 2>&1 && { echo "SERVER READY"; break; }
  kill -0 "${pid}" 2>/dev/null || { echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,150}" "${result_dir}/${label}-server."{out,err} 2>/dev/null | sort -u | head -4
    exit 1; }
  sleep 5
done
echo "--- backend selected ---"
grep -ohE "Using '[^']+' NvFp4 MoE backend[^.]*" "${result_dir}/${label}-server.out" | head -2
curl -fsS http://127.0.0.1:8031/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-nvfp4","prompt":"The capital of France is","max_tokens":20,"temperature":0}' | jq -r '.choices[0].text'
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
