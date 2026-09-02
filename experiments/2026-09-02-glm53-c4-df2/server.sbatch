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

# GLM-5.3 W4A16 (group 32) tiered MoE + DFlash2 at concurrency 4, DCP 1,
# 400K context, ~600K-token KV budget.
#
# DCP 1 is deliberate. DFlash2 loses ~38% of its acceptance under DCP4 for
# reasons still undiagnosed, so this ships the shape that measures well.
# GSM8K, 64 samples, eager draft:
#
#     DCP1 c=4   5.7046 AL, 136.2 tok/s     <- this config
#     DCP1 c=1   5.6450 AL, 140.3 tok/s
#     DCP4 c=4   3.5098 AL,  52.3 tok/s     <- do not ship
#
# Concurrency 4 is free here: at DCP1 it costs nothing against c=1 (5.7046 vs
# 5.6450, inside this harness's ~2% run-to-run spread).
#
# Differences from claude-glm53-c4.sh (MTP3, DCP4), all deliberate:
#   * speculator  DFlash2 (GLM-5.3-DFlash2), 7 speculative tokens, greedy draft
#                 sampling, FLASH_ATTN draft attention. Verifies 8 tokens per
#                 step against MTP3's 4, so the capture shapes are 8*1..8*4.
#   * dcp         1, not 4. See above.
#   * shape pin   VLLM_TIERED_MOE_RELAX_SHAPE=1, because the validator rejects
#                 max_num_seqs > 1 without DCP.
#   * kv budget   23,921,000,000 bytes per rank (22.28 GiB).
#
#     **600K tokens is not reachable on this shape.** A first attempt set
#     35,877,000,000 for ~600K and job 1626547 died in under three minutes
#     with "Fixed allocations and reserve exceed HBM capacity"
#     (tiered_moe_planner.py:328 -- hbm_capacity - reserve - weights - KV < 0).
#
#     At DCP 1 the target's MLA cache is replicated rather than sharded, so a
#     rank pays for whole sequences and KV is roughly 4x its DCP4 cost per
#     token. 22.28 GiB is the value a working DCP1 run (repl2-eager-dcp1)
#     auto-sized to, and it served 400K tokens; 33.4 GiB does not fit beside
#     the weights and the 6 GiB tiered reserve.
#
#     So the real choice on this node is:
#       DCP1 c=4  ~400K tokens  -- four concurrent sequences up to ~100K each,
#                                  or one at the full 400K. DFlash2 healthy.
#       DCP4 c=4  ~1.6M tokens  -- four full-length 400K sequences, but
#                                  DFlash2 loses ~38% acceptance there.
#     The tiered-MoE validator's "max_num_seqs > 1 requires DCP" is stating
#     exactly this memory fact. VLLM_TIERED_MOE_RELAX_SHAPE lifts the
#     assertion; it does not create the memory.
#
#     This trades against expert residency, which is HBM-bound: a DCP4 run
#     logged "2528 hot / 2272 cold experts per rank (50.0 GiB available /
#     20.3 MiB per expert)", so every GiB given to KV costs about 50 resident
#     experts per rank. Lower the budget if cold-hit rate matters more than
#     concurrency.

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
# max_num_seqs > 1 without DCP is rejected by the tiered-MoE validator, whose
# reasoning is that a replicated 400K MLA cache does not hold more than one
# sequence per rank. True for four *full-length* sequences; this runs a token
# budget instead, so four shorter ones fit. Lift the pin.
export VLLM_TIERED_MOE_RELAX_SHAPE=1
# DFlash2 verifies 8 tokens per step (7 drafted plus the bonus) where MTP3
# verifies 4, so the capture shapes are 8*1..8*4 at c=4.
#
# compile_sizes is EMPTY, unlike the MTP3 launcher's [4,8,12,16]. Every working
# DFlash2 arm runs it empty (2026-09-02-dflash-dcp-port/arm-dcp.sh:34), for two
# measured reasons. Inductor autotuning at these shapes asks for more shared
# memory than an SM has -- the first attempt at this file set
# compile_sizes=[8,16,24,32] and died at startup with
#   No valid triton configs. OutOfMemoryError: out of resource: triton_mm
#   Required: 262144  Hardware limit: 232448
# -- and separately a non-empty compile_sizes makes the DFlash2 selector a
# second compiled unit inside the draft's captured region, which aborts with
# "Expected exactly one compiled range_entry ... found 2" (see
# VLLM_DFLASH2_NO_SELECTOR_COMPILE).
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16,24,32],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

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
  --decode-context-parallel-size 1 \
  --max-num-seqs 4 \
  --host 0.0.0.0 \
  --port "${port}" \
  --gpu-memory-utilization 0.94 \
  --kv-cache-memory 23921000000 \
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
