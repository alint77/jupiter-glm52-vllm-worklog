#!/usr/bin/env bash
# DFlash2 on stock upstream vLLM, two nodes, no fork and no tiered MoE.
#
# This is the control that decides where the remaining DFlash2 acceptance gap
# lives. Upstream main carries DFlash2 (#52816) *and* every prerequisite that
# landed between this fork's 2026-07-16 base and DFlash2's 2026-08-20 merge,
# so it is the configuration the model card's numbers were produced against,
# modulo hardware.
#
#   upstream reaches ~5.94 on GSM8K -> our backport is still missing something
#   upstream also lands near 4.0    -> the gap is environmental (GH200 with
#                                      FLASH_ATTN against the card's GB300 with
#                                      FlashAttention 4), not our code
#
# PP2 x TP4/EP4 puts 54 GiB on each of 8 GPUs, so the 433 GiB checkpoint fits
# with no offload and no tiered code in the path at all.

set -euo pipefail

label="${1:?label}"
head_node="${2:?head node hostname}"
task="${3:-gsm8k}"          # gsm8k | humaneval
mode="${4:-dflash2}"        # dflash2 | mtp7 | none

up_dir=/e/project1/profound/alint77/vllm-upstream
result_dir=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-08-28-nvfp4-tiered

cd "${up_dir}"
# Modules only; the fork's env script would activate the fork's venv.
module load Stages/2026 >/dev/null 2>&1 || true
module load GCC/14.3.0 CUDA/13 CMake/3.31.8 NCCL/default-CUDA-13 >/dev/null 2>&1 || true
source "${up_dir}/.venv/bin/activate"
export TRITON_PTXAS_PATH="$(command -v ptxas)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "node:  $(hostname -s)   head: ${head_node}"
echo "label: ${label}  task: ${task}  mode: ${mode}"
echo "vllm:  $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-upstream"
export VLLM_SERVER_DEV_MODE=1
mkdir -p "${VLLM_CACHE_ROOT}"

export VLLM_HOST_IP="$(hostname -I | awk '{print $1}')"

if [[ "$(hostname -s)" == "${head_node}" ]]; then
  ray start --head --port=6380 --num-gpus=4 --disable-usage-stats \
    >"${result_dir}/${label}-rayhead.log" 2>&1
else
  sleep 20
  ray start --address="${head_node}:6380" --num-gpus=4 --disable-usage-stats \
    >"${result_dir}/${label}-rayworker.log" 2>&1
  sleep infinity
fi

# `ray status` text only: a client session would collide with vLLM's own.
for _ in $(seq 1 60); do
  gpus=$(ray status 2>/dev/null | sed -n 's@.*\([0-9]\+\)\.0*/\([0-9]\+\)\.0* GPU.*@\2@p' | head -1)
  [[ -z "${gpus}" ]] && gpus=0
  [[ "${gpus}" -ge 8 ]] && { echo "ray cluster ready: ${gpus} GPUs"; break; }
  sleep 5
done
[[ "${gpus:-0}" -ge 8 ]] || { echo "ray cluster never reached 8 GPUs (saw ${gpus:-0})"; exit 1; }

spec_args=()
case "${mode}" in
  dflash2) spec_args=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/e/project1/profound/alint77/models/GLM-5.3-DFlash2\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\"}") ;;
  mtp7)    spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":7}') ;;
esac

vllm serve "${model}" \
  --served-model-name glm53-upstream \
  --host 0.0.0.0 --port 8034 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --kv-cache-dtype fp8_ds_mla --block-size 64 \
  --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  "${spec_args[@]}" \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

ready=false
for _ in $(seq 1 500); do
  curl -fsS http://127.0.0.1:8034/health >/dev/null 2>&1 && { ready=true; break; }
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
echo "server ready (upstream, PP2 x TP4, ${mode})"

case "${task}" in
  gsm8k)     ds=/e/project1/profound/alint77/models/datasets/gsm8k/test.jsonl; field=question ;;
  humaneval) ds=/e/project1/profound/alint77/models/datasets/evalcache/humaneval.jsonl; field=prompt ;;
esac

# Acceptance, on the card's protocol.
python "${result_dir}/replicate_dflash2_eval.py" --label "${label}" \
  --out "${result_dir}/${label}-replicate.json" --num-samples 64 \
  --model glm53-upstream --dataset "${ds}" --prompt-field "${field}" || true

# Accuracy, five-shot, only meaningful for GSM8K.
if [[ "${task}" == gsm8k ]]; then
  export TMPDIR=/e/project1/profound/alint77/models/datasets/evalcache
  python "${up_dir}/tests/evals/gsm8k/gsm8k_eval.py" \
    --host http://127.0.0.1 --port 8034 --num-shots 5 --max-tokens 256 \
    --temperature 0 --seed 42 --num-questions 256 \
    --save-results "${result_dir}/${label}-gsm8k.json" 2>&1 | tail -6 || true
fi

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || true
