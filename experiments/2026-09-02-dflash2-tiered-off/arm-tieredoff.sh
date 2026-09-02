#!/usr/bin/env bash
# The card's protocol on the card's own checkpoint pair, with this fork's
# tiered MoE path OFF: native UVA expert offload instead. Everything else --
# checkpoint, drafter, width, sampling, dataset, sample count -- matches
# 2026-08-28-nvfp4-tiered/arm-replicate.sh, so the numbers are directly
# comparable to Phase 43-49e.
#
# Tiering off also lifts the tiered validator's pins (max_model_len=400000,
# max_num_batched_tokens=8192), so max_model_len drops to 16384: the eval
# never exceeds ~4.5K, and the 21 GB MLA reservation is HBM we need for
# weights here.
set -euo pipefail
label="${1:?label}"; mode="${2:-dflash2}"; offload="${3:-40}"; task="${4:-gsm8k}"
case "${task}" in
  gsm8k)     ds=/e/project1/profound/alint77/models/datasets/gsm8k/test.jsonl; field=question ;;
  humaneval) ds=/e/project1/profound/alint77/models/datasets/evalcache/humaneval.jsonl; field=prompt ;;
  *) echo "unknown task ${task}"; exit 1 ;;
esac
repo_dir=/e/project1/profound/alint77/vllm
script_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
result_dir="${RESULT_DIR:-${repo_dir}/agent_space/experiments/2026-09-02-dflash2-tiered-off/results}"
mkdir -p "${result_dir}"
cd "${repo_dir}"; source agent_space/jupiter-env.sh
echo "node: $(hostname)"; echo "label: ${label}  mode: ${mode}  offload_gb: ${offload}"
echo "tiered moe: OFF (native uva expert offload)"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
# One cache root for this (model, shape), not one per arm: per-arm roots are
# what exhausted the project1 inode quota on 2026-08-01.
export VLLM_CACHE_ROOT="/e/fscratch/profound/naeimitabiei1/caches/marlin/vllm-cache-nvfp4-tieredoff"
export TRTLLM_DG_CACHE_DIR="/e/fscratch/profound/naeimitabiei1/caches/marlin/trtllm-dg-nvfp4-tieredoff"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

if [[ "${mode}" == dflash2 ]]; then
  width=8; compile_sizes=""
  spec="{\"method\":\"dflash\",\"model\":\"/e/project1/profound/alint77/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"${DRAFT_SAMPLE_METHOD:-greedy}\"}"
else
  width=8; compile_sizes="8"
  spec='{"method":"mtp","num_speculative_tokens":7}'
fi

"${VLLM_VENV_DIR:-${PWD}/.venv}/bin/vllm" serve "${model}" \
  --served-model-name glm53-nvfp4-tieredoff \
  --host 127.0.0.1 --port 8027 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --distributed-executor-backend mp \
  --numa-bind \
  --offload-backend uva \
  --cpu-offload-gb "${offload}" \
  --cpu-offload-params experts \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 2 \
  --decode-context-parallel-size 1 \
  --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${width}],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}" \
  --speculative-config "${spec}" \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!
# A dying server answers /health long enough to fool the next arm (Phase 48).
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 || break
  sleep 2
done
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,140}" "${result_dir}/${label}-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -3; exit 1; }
  sleep 5
done
echo "server ready"
.venv/bin/python "${script_dir}/replicate_dflash2_eval.py" \
  --label "${label}" --out "${result_dir}/${label}-replicate.json" --num-samples 64 \
  --dataset "${ds}" --prompt-field "${field}"
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
