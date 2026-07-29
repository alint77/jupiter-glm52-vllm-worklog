#!/usr/bin/env bash
# Matched c1/q4 A/B of the tight-shared-memory Marlin launch.
#
#   sbatch job-qualify.sh off   # upstream launch (control)
#   sbatch job-qualify.sh on    # tight smem, hot 2 CTAs/SM + cold 1 CTA/SM
#
# Same binary for both arms; only VLLM_TIERED_MOE_TIGHT_SMEM differs. The
# configuration is copied from the routing-capture control arm, whose measured
# result is 99.78 output tok/s.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

label="${1:?arm is required: on|off}"
case "${label}" in
  on) tight=1 ;;
  off) tight=0 ;;
  *) echo "Unknown arm: ${label} (expected on|off)" >&2; exit 2 ;;
esac

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

unset VLLM_USE_V2_MODEL_RUNNER
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-marlin-smem"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-marlin-smem"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export VLLM_TIERED_MOE_TIGHT_SMEM="${tight}"
export TIERED_MOE_HBM_RESERVE_GB=5
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  >"${result_dir}/${label}-server.out" \
  2>"${result_dir}/${label}-server.err" &
server_pid=$!
monitor_pid=
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 720); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -80 "${result_dir}/${label}-server.err"
    exit 1
  fi
  sleep 10
done
[[ "${ready}" == true ]]

# Correctness gate: greedy eight-token continuation must be
#   " Paris. Distance from Paris to Lyon is"
curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}' \
  >"${result_dir}/${label}-smoke.json"
python3 -c "
import json, sys
text = json.load(open('${result_dir}/${label}-smoke.json'))['choices'][0]['text']
print('smoke continuation:', repr(text))
sys.exit(0 if text == ' Paris. Distance from Paris to Lyon is' else 1)
"

nvidia-smi \
  --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=2 \
  >"${result_dir}/${label}-vram-timeseries.csv" &
monitor_pid=$!

bench_args=(
  --backend openai
  --base-url http://127.0.0.1:8027
  --endpoint /v1/completions
  --model glm52-w4a16-tiered
  --served-model-name glm52-w4a16-tiered
  --tokenizer "${model}"
  --dataset-name custom
  --dataset-path "${prompts}"
  --custom-output-len 256
  --skip-chat-template
  --disable-shuffle
  --num-prompts 24
  --max-concurrency 1
  --request-rate inf
  --temperature 0
  --ignore-eos
  --disable-tqdm
)

.venv/bin/vllm bench serve "${bench_args[@]}"
curl -fsS http://127.0.0.1:8027/metrics >"${result_dir}/${label}-metrics-before.txt"
for repeat in 1 2; do
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${label}-r${repeat}.json"
done
curl -fsS http://127.0.0.1:8027/metrics >"${result_dir}/${label}-metrics-after.txt"
