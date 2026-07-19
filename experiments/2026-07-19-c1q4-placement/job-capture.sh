#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-19-c1q4-placement/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-19-c1q4-placement/slurm-%x-%j.err

set -euo pipefail

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-19-c1q4-placement
model=/e/project1/profound/alint77/models/GLM-5.2-W4A16-FP8-MTP
trace_dir="${result_dir}/trace-${SLURM_JOB_ID}"

export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/e/scratch/profound/naeimitabiei1/flashinfer-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-deepgemm-${SLURM_JOB_ID}"
export TIERED_MOE_HBM_RESERVE_GB=10

agent_space/experiments/2026-07-18-dcp-port/run-server.sh \
  3 1 FULL_AND_PIECEWISE "" --enable-return-routed-experts \
  >"${result_dir}/capture-${SLURM_JOB_ID}-server.out" \
  2>"${result_dir}/capture-${SLURM_JOB_ID}-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done
[[ "${ready}" == true ]]

.venv/bin/python agent_space/benchmarks/capture_routing_trace.py \
  --base-url http://127.0.0.1:8027 \
  --model glm52-w4a16-tiered \
  --prompt-file "${result_dir}/prompts.jsonl" \
  --tokenizer "${model}" \
  --output-dir "${trace_dir}" \
  --output-len 256 \
  --verification-size 4

for penalty in none 0.5 1.0 2.0; do
  extra=()
  suffix=per-expert
  if [[ "${penalty}" != none ]]; then
    extra=(--mixed-layer-penalty "${penalty}")
    suffix="hybrid-p${penalty}"
  fi
  .venv/bin/python agent_space/benchmarks/optimize_routing_profile.py \
    --trace-dir "${trace_dir}" \
    --model "${model}" \
    --hot-slots-per-rank 2870 \
    --train-requests 16 \
    --output-profile "${result_dir}/${suffix}-profile.json" \
    --output-report "${result_dir}/${suffix}-report.json" \
    "${extra[@]}"
done
