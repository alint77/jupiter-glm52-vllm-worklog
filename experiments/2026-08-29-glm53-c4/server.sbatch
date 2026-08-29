#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm53-claude-w4a16-c4
#SBATCH --output=agent_space/experiments/2026-08-29-glm53-c4/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-08-29-glm53-c4/slurm-%j.err

# GLM-5.3 W4A16 (group 32) tiered MoE, c=4 + MTP3 -- the 5.3 counterpart of
# the 5.2 AutoRound c4 launcher.
#
# This configuration is measured, not assumed. Three paired runs against the
# NVFP4 checkpoint on the real-code decode suite (512-token code prompts,
# 1024-token generations, c4/DCP4/MTP3):
#
#     W4A16 g32   213.1 +/- 1.4 tok/s aggregate, 18.77 ms TPOT, 477 ms TTFT
#     NVFP4 g16   194.2 +/- 4.6 tok/s aggregate, 20.60 ms TPOT, 479 ms TTFT
#
# +9.77% +/- 3.32%, same sign in all three pairs. TTFT is unchanged, so the win
# is decode. The likely mechanism is that this checkpoint quantizes the MTP
# block where NVFP4 leaves it BF16: 43.2 GB of non-routed weight against 57.1,
# about 3.5 GB per rank returned to the resident expert set.
#
# **Accuracy is NOT yet validated on this checkpoint.** NVFP4 passed GSM8K
# five-shot at 91.21%; W4A16 has no equivalent number. Group 32 with
# `observer: memoryless_minmax` is a coarser calibration. Roll back by pointing
# model_name at GLM-5.3-NVFP4 and the profile at
# agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json.
#
# Differences from the 5.2 script, all deliberate:
#   * checkpoint  JANGQ-AI/GLM-5.3-W4A16, compressed-tensors int4 group 32
#   * profile     glm53-w4a16-2496.json -- 9984 hot experts, 3940 replicas,
#                 carrying the GLM-5.3 ranking re-derived from an agentic
#                 coding routing capture (129,392 routed positions). That
#                 ranking is worth +8.3% over the 5.2 placeholder it replaced.
#                 2496 hot slots per rank, not 2400: gpu_memory_utilization
#                 0.94 with KV pinned buys 96 more experts per rank, taking
#                 held-out cold-hit from 0.2290 to 0.2131.
#   * reasoning   glm45 per the GLM-5.3 model card (tool stays glm47)
#   * no --tiered-moe-replica-assignment exact: the 5.2 script uses it, but it
#     was never exercised on 5.3 here, so it is left off rather than shipped
#     untested. Worth trying once and adding if it holds.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/${USER:-$(id -un)}/claude-glm53-c4}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-08-29-glm53-c4}"
[[ -w "${log_dir}" ]] || log_dir="${run_dir}/logs"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
# Caches on fscratch: GPFS is slow for the many small files these produce.
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-glm53-c4
export TRTLLM_DG_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/trtllm-dg-glm53-c4
# jupiter-env.sh points XDG_CACHE_HOME at /e/scratch (GPFS), and neither it nor
# this script set the inductor/triton dirs, so torch.compile artefacts landed on
# GPFS -- the filesystem that made the uv cache hang for an hour. Pin them to
# fscratch so a second launch actually reuses them.
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/triton
export TORCHINDUCTOR_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/inductor
export FLASHINFER_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/flashinfer

# Serve from fscratch: it scales with parallel readers where project1 does not
# (11 -> 21 GB/s across 4-8 readers vs 0.55 -> 0.95). Falls back to project1 so
# a scratch purge degrades startup instead of breaking the launch.
# W4A16 exists only on fscratch, so there is no project1 fallback to take:
# fail loudly rather than silently serving the wrong checkpoint, since the
# placement profile is fingerprinted to this one and would be rejected anyway.
model_name=GLM-5.3-W4A16
model_fscratch="/e/fscratch/profound/${USER:-$(id -un)}/models/${model_name}"
expected_shards=282
actual_shards="$(ls "${model_fscratch}"/*.safetensors 2>/dev/null | wc -l)"
if [[ ! -d "${model_fscratch}" ]] || (( actual_shards != expected_shards )); then
  echo "model: ${model_fscratch} missing or incomplete" \
       "(${actual_shards}/${expected_shards} shards)" >&2
  exit 1
fi
export TIERED_MOE_MODEL_PATH="${model_fscratch}"
echo "model: ${TIERED_MOE_MODEL_PATH} (${actual_shards} shards)"

export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
# 6 GB, not 7: at gpu_memory_utilization 0.94 the reserve check needs 4.66 GiB
# physically free after KV, and 7 GB would demand 5.59 GiB and refuse to start.
export TIERED_MOE_HBM_RESERVE_GB=6
# MTP3 verifies 4 tokens per step, so the graph shapes are 4*1..4*4 at c=4.
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

mkdir -p "${run_dir}" "${log_dir}" "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
rm -f "${ready_file}" "${endpoint_file}"

server_pid=
cleanup() {
  rm -f "${ready_file}" "${endpoint_file}"
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --max-num-seqs 4 \
  --host 0.0.0.0 \
  --port "${port}" \
  --gpu-memory-utilization 0.94 \
  --kv-cache-memory 21689598771 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --generation-config vllm \
  --served-model-name glm53-w4a16-tiered \
  >"${log_dir}/server-${SLURM_JOB_ID}.out" \
  2>"${log_dir}/server-${SLURM_JOB_ID}.err" &
server_pid=$!

ready=false
for _ in $(seq 1 240); do
  if curl -fsS -H "Authorization: Bearer ${api_key}" \
      "http://127.0.0.1:${port}/health" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done
[[ "${ready}" == true ]]

umask 077
node="$(hostname -s)"
tmp="${endpoint_file}.tmp"
printf 'JOB_ID=%q\nNODE=%q\nPORT=%q\n' \
  "${SLURM_JOB_ID}" "${node}" "${port}" >"${tmp}"
mv "${tmp}" "${endpoint_file}"
touch "${ready_file}"

wait "${server_pid}"
