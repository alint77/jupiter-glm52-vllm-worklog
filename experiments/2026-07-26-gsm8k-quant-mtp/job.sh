#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-26-gsm8k-quant-mtp/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-26-gsm8k-quant-mtp/slurm-%x-%j.err

set -euo pipefail

label="${1:?configuration label is required}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-26-gsm8k-quant-mtp"
dataset_dir="$(dirname -- "${repo_dir}")/models/datasets/gsm8k"

case "${label}" in
  w4-target)
    model="$(dirname -- "${repo_dir}")/models/GLM-5.2-W4A16-FP8-MTP"
    profile="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/hybrid-p0.5-profile.json"
    depth=0
    ;;
  w4-mtp3)
    model="$(dirname -- "${repo_dir}")/models/GLM-5.2-W4A16-FP8-MTP"
    profile="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/hybrid-p0.5-profile.json"
    depth=3
    ;;
  autoround-target)
    model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
    profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
    depth=0
    ;;
  autoround-mtp3)
    model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
    profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
    depth=3
    ;;
  *)
    echo "Unknown configuration: ${label}" >&2
    exit 2
    ;;
esac

cd "${repo_dir}"
source agent_space/jupiter-env.sh

cp "${dataset_dir}/train.jsonl" /tmp/train.jsonl
cp "${dataset_dir}/test.jsonl" /tmp/test.jsonl

verification_size=$((depth + 1))
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-dg-${SLURM_JOB_ID}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=1
export TIERED_MOE_HBM_RESERVE_GB=10
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${verification_size}],\"compile_sizes\":[${verification_size}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

server_args=(--no-enable-prefix-caching)
if ((depth > 0)); then
  server_args+=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${depth}}"
  )
fi

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  "${server_args[@]}" \
  >"${result_dir}/${label}-server.out" \
  2>"${result_dir}/${label}-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

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

eval_args=(
  --host http://127.0.0.1
  --port 8027
  --num-shots 5
  --max-tokens 256
  --temperature 0
  --seed 42
)

.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
  "${eval_args[@]}" \
  --num-questions 8 \
  >"${result_dir}/${label}-warmup.out"

curl -fsS http://127.0.0.1:8027/metrics \
  >"${result_dir}/${label}-metrics-before.txt"

for repeat in 1 2; do
  .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
    "${eval_args[@]}" \
    --num-questions 256 \
    --save-results "${result_dir}/${label}-r${repeat}.json" \
    >"${result_dir}/${label}-r${repeat}.out"
done

curl -fsS http://127.0.0.1:8027/metrics \
  >"${result_dir}/${label}-metrics-after.txt"
