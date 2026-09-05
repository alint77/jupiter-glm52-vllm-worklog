#!/usr/bin/env bash
# Torch-profiler captures of prefill and decode for MTP3 at c=1, the shape the
# 2026-09-04 speculator comparison recommends serving Claude Code with.
# This run profiles the PRODUCTION checkpoint (GLM-5.2 AutoRound W4G64, the
# auto_gptq path) with the prefetch on, as server.sbatch now enables it.
# The campaign's measurements were all GLM-5.3 W4A16 group 32; group 64 has
# a different scale-to-weight byte ratio, so the open question is whether
# staged cold Marlin still lands at the hot tier's rate.
#
# Prefix caching is off here, as in every profiling run, so a repeated
# prompt cannot skip the prefill being measured. Production runs it ON, so
# a real session with prefix hits sees smaller chunks and less benefit.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=prodprof
#SBATCH --output=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
# Prompts and capture.py belong to the profiling campaign; this run's
# outputs belong beside the phase worklogs.
input_dir="${repo_dir}/agent_space/experiments/2026-09-04-mtp3-profile"
result_dir="${repo_dir}/agent_space/experiments/2026-09-05-cold-prefetch"
prompts="${input_dir}/prompts.jsonl"
trace_root="/e/project1/profound/alint77/traces/prod-prefetch-${SLURM_JOB_ID}"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

model_name=GLM-5.2-AutoRound-W4G64-MTP-e1ba887
model="/e/fscratch/profound/${USER:-$(id -un)}/models/${model_name}"
[[ -d "${model}" ]] || model="/e/project1/profound/alint77/models/${model_name}"
[[ -d "${model}" ]] || { echo "model missing: ${model}" >&2; exit 1; }

mkdir -p "${trace_root}"
printf '%s\n' "${trace_root}" >"${result_dir}/trace-path-${SLURM_JOB_ID}.txt"
echo "node: $(hostname)"
echo "trace root: ${trace_root}"

# The serving configuration unchanged, except that prefix caching is off so a
# repeated prompt cannot skip the prefill this is meant to measure.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export TIERED_MOE_HBM_RESERVE_GB=5
# The feature under measurement.
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024
export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=0
cache_root="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin"
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-prodprof"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-prodprof"
export TRITON_CACHE_DIR="${cache_root}/triton-prodprof"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-prodprof"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-prodprof"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
# MTP3 verifies 4 tokens per step; compile the shape that is captured.
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

# record_shapes drives the kernel-shape attribution the analysis needs;
# with_stack is off because Python stacks on a 96K prefill make the trace
# unloadable. ignore_frontend keeps the API server's own work out.
profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_root}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 1 \
  --max-num-seqs 1 \
  --served-model-name glm53-cmp-tiered \
  --gpu-memory-utilization 0.90 \
  --max-model-len 400000 \
  --no-enable-prefix-caching \
  --profiler-config "${profiler_config}" \
  >"${result_dir}/prod-profile-server.out" \
  2>"${result_dir}/prod-profile-server.err" &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT

ready=false
for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    grep -ohE "CUDA out of memory[^\"]{0,140}|[A-Za-z]*(Error|Exception): .{0,140}" \
      "${result_dir}/prod-profile-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -3
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]]
echo "server ready"

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-cmp-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  >"${result_dir}/prod-profile-semantic.json"

.venv/bin/python "${input_dir}/capture.py" \
  --prompts "${prompts}" \
  --trace-root "${trace_root}"

cp "${trace_root}/capture-report.json" "${result_dir}/capture-report-${SLURM_JOB_ID}.json"
du -sh "${trace_root}"/* | tee "${result_dir}/trace-sizes-${SLURM_JOB_ID}.txt"
echo "=== done ==="
