#!/usr/bin/env bash
# Torch-profiler captures of prefill and decode on the production configuration.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=prod-profile
#SBATCH --output=agent_space/experiments/2026-08-05-prod-profile/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-prod-profile/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-05-prod-profile"
prompts="${repo_dir}/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/prompts.jsonl"
trace_root="/e/project1/profound/alint77/traces/prod-profile-${SLURM_JOB_ID}"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
[[ -d "${model}" ]] || model="${repo_dir}/../models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"

mkdir -p "${trace_root}"
printf '%s\n' "${trace_root}" >"${result_dir}/trace-path-${SLURM_JOB_ID}.txt"
echo "node: $(hostname)"
echo "trace root: ${trace_root}"

# Production configuration, unchanged except for the profiler and prefix
# caching, which is disabled so a repeated prompt cannot skip its prefill.
# Production runs the V2 model runner; its complete MTP CUDA graphs are the
# whole point of that choice at c4, so profiling V1 here would measure a
# configuration nobody serves.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-prod-profile"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-prod-profile"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-31-replica-scheduling-v2/hybrid-p0.5-replicas-985.json"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_root}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --tiered-moe-replica-assignment exact \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --profiler-config "${profiler_config}" \
  >"${result_dir}/profile-server.out" \
  2>"${result_dir}/profile-server.err" &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT

ready=false
for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -80 "${result_dir}/profile-server.err"
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]]

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  >"${result_dir}/semantic.json"

.venv/bin/python "${result_dir}/capture.py" \
  --prompts "${prompts}" \
  --trace-root "${trace_root}"

cp "${trace_root}/capture-report.json" "${result_dir}/capture-report-${SLURM_JOB_ID}.json"
du -sh "${trace_root}"/* | tee "${result_dir}/trace-sizes-${SLURM_JOB_ID}.txt"
echo "=== done ==="
