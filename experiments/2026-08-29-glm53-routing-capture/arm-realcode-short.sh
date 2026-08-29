#!/usr/bin/env bash
# GLM-5.3 NVFP4 on the production 5.2 configuration: MTP3, c4, DCP4, and the
# same 16K-in / 512-out PyTorch coding suite Phase 33 measured, so the numbers
# line up against 265.2 tok/s decode aggregate and 70.98 end to end at c4.
#
# **The placement ranking is a GLM-5.2 placeholder.** 5.3's post-training moved
# the router, so hot-expert residency here is wrong in a way that costs decode
# throughput and nothing else. Read these numbers as a floor, not as the
# configuration's performance.
#
# MTP3 instantiates layer 78, whose experts are BF16 in this checkpoint: 4.50
# GiB per rank against W4G64's 1.21. That is 3.29 GiB the profile has to leave
# free on top of everything else. DFlash2 does not instantiate layer 78 at
# all, so that cost disappears and its own ~4.58 GiB draft, TP-sharded, is
# cheaper than the head it replaces.

set -euo pipefail

label="${1:?label}"
concurrency="${2:-4}"
profile="${3:?placement profile}"
mode="${4:-mtp3}"   # mtp3 | dflash2

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
bench_dir="${repo_dir}/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node:  $(hostname)"
head_ref="$(<"${repo_dir}/.git/HEAD")"
branch="${head_ref#ref: refs/heads/}"
echo "commit: $(<"${repo_dir}/.git/refs/heads/${branch}") (${branch})"
echo "label: ${label}  concurrency: ${concurrency}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/fscratch/profound/naeimitabiei1/caches/marlin/vllm-cache-nvfp4"
export TRTLLM_DG_CACHE_DIR="/e/fscratch/profound/naeimitabiei1/caches/marlin/trtllm-dg-nvfp4"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-7}"

# Verification width: 4 under MTP3 (3 draft + bonus), 8 under DFlash2 (block 8).
if [[ "${mode}" == dflash2 ]]; then
  width=8
  draft=/e/project1/profound/alint77/models/GLM-5.3-DFlash2
  spec_config="{\"method\":\"dflash\",\"model\":\"${draft}\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"}"
  # DFlash2's CandidateSelector takes only static shapes, so piecewise_backend
  # uses its static-shape branch and asserts exactly one compiled range entry.
  # A non-empty compile_sizes gives it two. See the Phase 42 backport.
  compile_sizes=""
else
  width=4
  spec_config='{"method":"mtp","num_speculative_tokens":3}'
  compile_sizes_set=1
fi
sizes=""
for ((i = 1; i <= concurrency; i++)); do sizes+="${sizes:+,}$((width * i))"; done
[[ -n "${compile_sizes_set:-}" ]] && compile_sizes="${sizes}"
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${sizes}],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

dcp=1
[[ "${concurrency}" -gt 1 ]] && dcp=4

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${spec_config}" \
  --decode-context-parallel-size "${dcp}" \
  --max-num-seqs "${concurrency}" \
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

bench() {
  .venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered --tokenizer "${model}" \
    --dataset-name custom --dataset-path "/e/project1/profound/alint77/vllm/agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl" \
    --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
    --max-concurrency "${concurrency}" --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm "$@"
}

echo "=== warmup (excluded) ==="
bench >/dev/null 2>&1 || true

# The reference harness resets before every measured run. Without this the
# measured pass replays the warmup's prompts out of the prefix cache, which
# makes TTFT meaningless and inflates end-to-end throughput. TPOT is unharmed
# either way, but prefill is the point of measuring it.
curl -fsS -X POST http://127.0.0.1:8027/reset_prefix_cache >/dev/null \
  && echo "prefix cache reset" || echo "WARNING: prefix cache reset failed"

curl -fsS http://127.0.0.1:8027/metrics \
  | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total' \
  >"${result_dir}/${label}-spec-before.txt" || true

echo "=== measured run ==="
bench --save-result --save-detailed --result-dir "${result_dir}" \
      --result-filename "${label}-bench.json" 2>&1 | tail -30

curl -fsS http://127.0.0.1:8027/metrics \
  | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total' \
  >"${result_dir}/${label}-spec-after.txt" || true

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
