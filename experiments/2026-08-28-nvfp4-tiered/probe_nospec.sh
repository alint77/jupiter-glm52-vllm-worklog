#!/usr/bin/env bash
# Reproduce the no-speculator five-shot failure and CAPTURE the completions,
# which the vllm gsm8k eval discards. Runs the identical prompt three ways so
# the failing axis is unambiguous:
#   A. five-shot raw completion, max_tokens 256, stop sequences  (the failing mode)
#   B. same but no stop sequences
#   C. same but max_tokens 2048
set -euo pipefail
label="${1:?label}"; profile="${2:?profile}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"; source agent_space/jupiter-env.sh
model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-nvfp4"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-nvfp4"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[1],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
# NO speculator -- the failing configuration
agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --decode-context-parallel-size 1 --max-num-seqs 1 \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "SERVER EXITED EARLY"; exit 1; }
  sleep 5
done
echo "server ready (NO speculator)"
export TMPDIR=/e/project1/profound/alint77/models/datasets/evalcache
.venv/bin/python "${result_dir}/probe_nospec.py" --out "${result_dir}/${label}-probe.json"
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
