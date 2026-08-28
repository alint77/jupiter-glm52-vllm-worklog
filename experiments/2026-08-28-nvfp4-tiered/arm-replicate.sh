#!/usr/bin/env bash
# Serve, then run the DFlash2 card's acceptance protocol against it.
set -euo pipefail
label="${1:?label}"; mode="${2:-dflash2}"; profile="${3:?profile}"
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

if [[ "${mode}" == dflash2 ]]; then
  width=8; compile_sizes=""
  spec="{\"method\":\"dflash\",\"model\":\"/e/project1/profound/alint77/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"}"
else
  # The card's MTP baseline also proposes seven tokens, not three.
  width=8; compile_sizes="8"
  spec='{"method":"mtp","num_speculative_tokens":7}'
fi
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${width}],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${spec}" --decode-context-parallel-size 1 --max-num-seqs 1 \
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
.venv/bin/python "${result_dir}/replicate_dflash2_eval.py" \
  --label "${label}" --out "${result_dir}/${label}-replicate.json" --num-samples 64
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
