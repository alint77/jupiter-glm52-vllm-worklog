#!/usr/bin/env bash
# GSM8K accuracy + losslessness check. mode: none | mtp7 | dflash2
set -euo pipefail
label="${1:?label}"; mode="${2:-none}"; profile="${3:?profile}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"; source agent_space/jupiter-env.sh
echo "node: $(hostname)"; echo "label: ${label}  mode: ${mode}"
model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-nvfp4"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-nvfp4"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export TIERED_MOE_HBM_RESERVE_GB=7
spec_args=()
case "${mode}" in
  dflash2) w=8; cs=""
    spec_args=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/e/project1/profound/alint77/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"}") ;;
  mtp7)    w=8; cs="8"
    spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":7}') ;;
  none)    w=1; cs="1" ;;
esac
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${w}],\"compile_sizes\":[${cs}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  "${spec_args[@]}" --decode-context-parallel-size 1 --max-num-seqs 1 \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,140}" "${result_dir}/${label}-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -3; exit 1; }
  sleep 5
done
echo "server ready"
.venv/bin/python "${result_dir}/score_gsm8k.py" --label "${label}" \
  --out "${result_dir}/${label}-score.json" --num-samples 64
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
