#!/usr/bin/env bash
# Serve, then run the DFlash2 card's acceptance protocol against it.
set -euo pipefail
label="${1:?label}"; mode="${2:-dflash2}"; profile="${3:?profile}"; task="${4:-gsm8k}"
case "${task}" in
  gsm8k)     ds=/e/fscratch/profound/naeimitabiei1/models/datasets/gsm8k/test.jsonl; field=question ;;
  humaneval) ds=/e/fscratch/profound/naeimitabiei1/models/datasets/evalcache/humaneval.jsonl; field=prompt ;;
  # 16K PyTorch source prompts. The draft's sliding window is 2048, so every
  # one of these evicts context -- which is the only shape that exercises the
  # null-block guard. GSM8K and HumanEval are short enough never to evict.
  longcode)  ds=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/prompts.jsonl; field=prompt ;;
  *) echo "unknown task ${task}"; exit 1 ;;
esac
repo_dir=/e/project1/profound/alint77/vllm
# Where the harness lives, which is not where results go once RESULT_DIR
# redirects them.
script_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
result_dir="${RESULT_DIR:-${script_dir}}"
cd "${repo_dir}"; source agent_space/jupiter-env.sh
echo "node: $(hostname)"; echo "label: ${label}  mode: ${mode}"
echo "draft rope: ${VLLM_DFLASH_DRAFT_ROPE:-neox}"
echo "null-block guard: ${VLLM_DFLASH_NULL_BLOCK_GUARD:-1}"
echo "draft sample method: ${DRAFT_SAMPLE_METHOD:-greedy}"
model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/fscratch/profound/naeimitabiei1/caches/marlin/vllm-cache-nvfp4"
export TRTLLM_DG_CACHE_DIR="/e/fscratch/profound/naeimitabiei1/caches/marlin/trtllm-dg-nvfp4"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export TIERED_MOE_HBM_RESERVE_GB=7

if [[ "${mode}" == dflash2 ]]; then
  width=8; compile_sizes=""
  # draft_sample_method: "greedy" (default) makes the selector take an argmax
  # walk, so the lattice's proposal distribution q never reaches the verifier
  # and the ratio test degenerates to p(argmax q). SGLang samples the path and
  # passes q. DRAFT_SAMPLE_METHOD selects between them.
  dsm="${DRAFT_SAMPLE_METHOD:-greedy}"
  spec="{\"method\":\"dflash\",\"model\":\"/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"${dsm}\"}"
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
# Wait for any previous arm's server to stop answering before trusting /health:
# two arms in one allocation share port 8027, and a dying server answers long
# enough to make the next arm declare readiness and then measure the wrong
# build. Bounded, so a genuinely fast start is not penalised.
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
