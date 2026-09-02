#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm53-claude-w4a16-c4-df2
#SBATCH --output=agent_space/experiments/2026-09-02-glm53-c4-df2/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-02-glm53-c4-df2/slurm-%j.err

# GLM-5.3 W4A16 (group 32) tiered MoE, c=4 + DFlash2 -- the DFlash2 counterpart
# of claude-glm53-c4.sh, which runs the same shape with MTP3.
#
# Differences from the MTP3 launcher, all deliberate:
#   * speculator  DFlash2 (GLM-5.3-DFlash2), 7 speculative tokens, greedy draft
#                 sampling, FLASH_ATTN draft attention. Verifies 8 tokens per
#                 step against MTP3's 4, so the graph shapes are 8*1..8*4.
#   * kv budget   11,773,200,000 bytes, sized for ~600K tokens.
#
#     Derived, not guessed. The MTP3 launcher's 21,689,598,771 bytes yielded
#     "GPU KV cache size: 1,600,255 tokens" -> 13,554 bytes/token. A DFlash2
#     DCP4 server reported needing 7.31 GiB for one 400K sequence ->
#     19,622 bytes/token. The 6,068 bytes/token difference is the drafter's
#     replicated KV (6 layers x 2 kv heads x 128 x 2 x bf16 = 6,144
#     bytes/token), which is the arithmetic agreeing with itself.
#     600,000 x 19,622 ~= 11.77e9. At 400K max_model_len that is 1.5
#     concurrent full-length sequences, or four shorter ones.
#
# **KNOWN DEFICIT -- READ BEFORE TRUSTING THIS.** DFlash2 acceptance under DCP4
# is materially worse than at DCP1 and the cause is not yet found. Measured on
# GSM8K, 64 samples, eager draft, 32K context:
#
#     DCP1 c=1   5.6450 AL      DCP1 c=4   5.7046 AL
#     DCP4 c=4   3.5098 AL
#
# So the DCP4 shape this script runs costs ~38% of acceptance. Ruled out by
# measurement, so do not re-investigate: the concurrency setting (DCP1 at c=4
# is clean), the drafter's slot mapping (0% PAD at both DCP sizes), the
# target's hidden states (MTP is flat at DCP4, 4.9920 vs 4.8762), and
# seq_lens substitution. See
# agent_space/experiments/2026-09-02-dflash-dcp-port/README.md.
#
# Until that is resolved, claude-glm53-c4.sh (MTP3) is the better production
# choice on this shape. This launcher exists to exercise DFlash2 at c=4 and to
# measure the deficit, not because it is currently faster.
#
# Also unvalidated: every DFlash2 number so far was measured against the
# GLM-5.3-NVFP4 target, while this script serves GLM-5.3-W4A16 to match the
# existing c=4 launcher. The drafter is quantization-independent in principle
# (its RoPE layout is pinned by VLLM_DFLASH_DRAFT_ROPE, not by the target's
# weights), but the pairing has not been measured.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/claude-glm53-c4-df2}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-09-02-glm53-c4-df2}"
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
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-glm53-c4-df2
export TRTLLM_DG_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/trtllm-dg-glm53-c4-df2
# jupiter-env.sh points XDG_CACHE_HOME at /e/fscratch, and neither it nor
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
# DFlash2 verifies 8 tokens per step (7 drafted plus the bonus) where MTP3
# verifies 4, so the graph shapes are 8*1..8*4 at c=4.
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16,24,32],"compile_sizes":[8,16,24,32],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

# The DFlash2 drafter, a sibling of the target checkpoint on fscratch.
drafter="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-DFlash2"
if [[ ! -d "${drafter}" ]]; then
  echo "drafter: ${drafter} missing" >&2
  exit 1
fi
echo "drafter: ${drafter}"

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
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}" \
  --decode-context-parallel-size 4 \
  --max-num-seqs 4 \
  --host 0.0.0.0 \
  --port "${port}" \
  --gpu-memory-utilization 0.94 \
  --kv-cache-memory 11773200000 \
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
