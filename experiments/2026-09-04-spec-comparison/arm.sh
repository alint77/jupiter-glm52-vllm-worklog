#!/usr/bin/env bash
# One comparison arm: serve the W4A16 tiered c=1 400K shape with ONE
# speculative config, then stream the CC-shaped corpus at it.
#
# All four arms share everything but the speculator -- model, profile,
# utilisation 0.90, reserve 10, capture [K+1] -- so the comparison isolates
# the speculative method and width. Reserve is today's proven 400K value
# (job 1659668); with the planner now budgeting draft pages, MTP has more
# margin than DFlash at the same reserve, so uniform 10 risks nothing.
#
# Jobs run on separate nodes, each a full allocation, so the port is shared
# harmlessly and the client measures against 127.0.0.1 with the wire removed.
set -euo pipefail
label="${1:?label}"   # df2-3 | df2-7 | mtp3 | mtp7
mode="${2:?mode}"     # dflash | mtp
width="${3:?K}"       # 3 | 7 (num_speculative_tokens; verify = K+1 per step)

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${here}/../../.." && pwd)"
cd "${repo_dir}"
source agent_space/jupiter-env.sh
result_dir="${RESULT_DIR:-${here}}"
mkdir -p "${result_dir}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16"
expected_shards=282
actual_shards="$(ls "${model}"/*.safetensors 2>/dev/null | wc -l)"
if (( actual_shards != expected_shards )); then
  echo "model: ${model} incomplete (${actual_shards}/${expected_shards})" >&2
  exit 1
fi

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=10
# Per-arm caches: four arms compile concurrently against GPFS-fscratch, and
# shared roots would race. The compile hash separates widths anyway, but
# separate roots also keep a failed arm from poisoning another's cache.
cache_root="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin"
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-spec-cmp-${label}"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-spec-cmp-${label}"
# jupiter-env.sh leaves the inductor/triton dirs on GPFS, which made a uv
# cache hang for an hour once. Pin them to fscratch, per.arm.
export TRITON_CACHE_DIR="${cache_root}/triton-spec-cmp-${label}"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-spec-cmp-${label}"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-spec-cmp-${label}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"

if [[ "${mode}" == "dflash" ]]; then
  drafter="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-DFlash2"
  compile_sizes=""
  spec="{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":${width},\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}"
else
  # K+1, matching cudagraph_capture_sizes: a step verifies the bonus token
  # plus K drafts. Every proven MTP config compiles the shape it captures --
  # the production MTP3 launcher is [4,8,12,16]/[4,8,12,16], and arm-dcp.sh
  # sets width=8 for K=7 and compiles "8". Compiling K instead specialises a
  # batch that never runs and leaves the real one on the dynamic path, which
  # would have penalised the MTP arms only.
  compile_sizes="$((width + 1))"
  spec="{\"method\":\"mtp\",\"num_speculative_tokens\":${width}}"
fi
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$((width + 1))],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

echo "=== ${label}: mode=${mode} K=${width} node=$(hostname) ==="

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${spec}" \
  --decode-context-parallel-size 1 \
  --max-num-seqs 1 \
  --served-model-name glm53-cmp-tiered \
  --gpu-memory-utilization 0.90 \
  --max-model-len 400000 \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

# Bounded port-wait, the arm-dcp pattern: a previous arm's server must stop
# answering before /health can be trusted, but a fast start is not penalised.
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 || break
  sleep 2
done
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "CUDA out of memory[^\"]{0,140}|[A-Za-z]*(Error|Exception): .{0,140}" \
      "${result_dir}/${label}-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -3
    exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8027/health >/dev/null
echo "server ready"

sleep 3
# 2048, not 1024: GLM-5.3 reasons before answering and a 700-token probe was
# still mid-thought, so a lower cap would compare reasoning throughput only
# and never reach the code or prose the tasks ask for.
.venv/bin/python "${here}/bench_client.py" \
  --label "${label}" --mode "${mode}" --width "${width}" \
  --max-tokens 2048 \
  --out "${result_dir}/${label}-result.json" \
  >"${result_dir}/${label}-bench.out" 2>&1 || {
    echo "BENCH CLIENT FAILED"; tail -5 "${result_dir}/${label}-bench.out"; exit 1;
  }
tail -3 "${result_dir}/${label}-bench.out"

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
echo "=== ${label} done ==="
