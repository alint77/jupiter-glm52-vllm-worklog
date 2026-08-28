#!/usr/bin/env bash
# GLM-5.3 NVFP4 on the project's own GSM8K gate: tests/evals/gsm8k/gsm8k_eval.py,
# five-shot, 256 questions, twice, exactly as Phase 25/29 ran it. Those runs
# produced the reference band this is measured against:
#   W4G128 target-only 93.55% | W4G128+MTP3 94.34%
#   AutoRound target-only 95.51% | AutoRound+MTP3 95.70%
#
# Five-shot completion format rather than chat, so answer extraction is exact.
# Writing my own zero-shot chat scorer was the mistake that produced a
# spurious 79.7%: the model emits markdown with \boxed{}, LaTeX and a closing
# sentence, and "last integer in the text" picks the wrong number.
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
  mtp3)    w=4; cs="4"; spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
  dflash2) w=8; cs=""
    spec_args=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/e/project1/profound/alint77/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"}") ;;
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
# gsm8k_eval.py fetches the dataset from GitHub and Booster nodes have no
# internet. download_and_cache_file checks os.path.exists(TMPDIR/<basename>)
# before fetching, so a pre-seeded TMPDIR satisfies it offline.
export TMPDIR=/e/project1/profound/alint77/models/datasets/evalcache
eval_args=(--host http://127.0.0.1 --port 8027 --num-shots 5 --max-tokens 256 --temperature 0 --seed 42)
.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py "${eval_args[@]}" --num-questions 8 >/dev/null 2>&1 || true
for repeat in 1 2; do
  echo "--- repeat ${repeat} ---"
  .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py "${eval_args[@]}" \
    --num-questions 256 --save-results "${result_dir}/${label}-r${repeat}.json" 2>&1 | tail -6
done
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
