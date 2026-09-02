#!/usr/bin/env bash
# Prefill + decode performance at concurrency 4, on the 16K-in / 1024-out
# PyTorch coding suite, with real concurrent load (vllm bench serve,
# --max-concurrency N --request-rate inf).
#
# Adapted from 2026-08-29-glm53-routing-capture/arm-realcode-short.sh with two
# changes that matter:
#
#   * DCP is an argument, not `[[ concurrency -gt 1 ]] && dcp=4`. DFlash2 loses
#     ~38% acceptance under DCP4 (3.5098 against 5.7046 at DCP1 c=4), so the
#     shape worth measuring is DCP1.
#   * W4A16 target and the glm53-w4a16-2496 profile, matching
#     claude-glm53-c4-df2.sh rather than the older NVFP4 arms.
#
# The acceptance harness used everywhere else in this directory issues requests
# sequentially, so it has never produced an aggregate-under-load number. This
# one does.
set -euo pipefail

label="${1:?label}"
mode="${2:-dflash2}"        # dflash2 | mtp3
concurrency="${3:-4}"
dcp="${4:-1}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-09-02-df2-c4-perf/results"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node: $(hostname)"
echo "label: ${label}  mode: ${mode}  concurrency: ${concurrency}  dcp: ${dcp}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16"
drafter="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-DFlash2"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-glm53-c4-df2"
export TRTLLM_DG_CACHE_DIR="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/trtllm-dg-glm53-c4-df2"
export TRITON_CACHE_DIR="/e/fscratch/profound/${USER:-$(id -un)}/caches/triton"
export TORCHINDUCTOR_CACHE_DIR="/e/fscratch/profound/${USER:-$(id -un)}/caches/inductor"
mkdir -p "${result_dir}" "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=6
# max_num_seqs > 1 without DCP is rejected by the tiered-MoE shape validator.
export VLLM_TIERED_MOE_RELAX_SHAPE=1

if [[ "${mode}" == dflash2 ]]; then
  width=8
  spec_config="{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}"
  # Must stay empty: a non-empty compile_sizes makes the DFlash2 selector a
  # second compiled unit in the captured region, and inductor autotuning at
  # these shapes also exceeds an SM's shared memory.
  compile_sizes=""
  kv_bytes="${KV_BYTES:-35877000000}"
else
  width=4
  spec_config='{"method":"mtp","num_speculative_tokens":3}'
  compile_sizes_set=1
  kv_bytes="${KV_BYTES:-21689598771}"
fi

sizes=""
for ((i = 1; i <= concurrency; i++)); do sizes+="${sizes:+,}$((width * i))"; done
[[ -n "${compile_sizes_set:-}" ]] && compile_sizes="${sizes}"
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${sizes}],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${spec_config}" \
  --decode-context-parallel-size "${dcp}" \
  --max-num-seqs "${concurrency}" \
  --kv-cache-memory "${kv_bytes}" \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

ready=false
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,150}" \
      "${result_dir}/${label}-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -4
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]] || { echo "SERVER NOT READY"; kill "${pid}" 2>/dev/null; exit 1; }
echo "server ready"
grep -ohE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency[^|]{0,45}|Tiered MoE residency:.*" \
  "${result_dir}/${label}-server.out" | sort -u | head -3

bench() {
  .venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered --tokenizer "${model}" \
    --dataset-name custom \
    --dataset-path "${repo_dir}/agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl" \
    --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
    --max-concurrency "${concurrency}" --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm "$@"
}

echo "=== warmup (excluded) ==="
bench >/dev/null 2>&1 || true
# Without this the measured pass replays the warmup's prompts out of the prefix
# cache, which makes TTFT meaningless. Prefill is half the point here.
curl -fsS -X POST http://127.0.0.1:8027/reset_prefix_cache >/dev/null \
  && echo "prefix cache reset" || echo "WARNING: prefix cache reset failed"

curl -fsS http://127.0.0.1:8027/metrics \
  | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total' \
  >"${result_dir}/${label}-spec-before.txt" || true

echo "=== measured run ==="
bench --save-result --save-detailed --result-dir "${result_dir}" \
      --result-filename "${label}-bench.json" 2>&1 | tail -32

curl -fsS http://127.0.0.1:8027/metrics \
  | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total' \
  >"${result_dir}/${label}-spec-after.txt" || true

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
