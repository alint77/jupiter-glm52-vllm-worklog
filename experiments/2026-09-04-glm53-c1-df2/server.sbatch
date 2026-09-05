#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=glm53-claude-w4a16-c1-df2
#SBATCH --output=agent_space/experiments/2026-09-04-glm53-c1-df2/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-04-glm53-c1-df2/slurm-%j.err

# GLM-5.3 W4A16 (group 32) tiered MoE + DFlash2 at concurrency 1, DCP 1,
# 400K context.
#
# The tiered-MoE production shape with nothing relaxed. All three shape pins
# in config/vllm.py hold natively at c=1 -- max_num_seqs > 1 without DCP
# (2330), max_num_batched_tokens != 8192 (2342), max_model_len != 400000
# (2346) -- so VLLM_TIERED_MOE_RELAX_SHAPE is absent here and the validator
# stays in force. The c=4 sibling has to lift it.
#
# DCP 1 is deliberate. DFlash2 loses ~38% of its acceptance under DCP4 for
# reasons still undiagnosed, so this ships the shape that measures well.
#
# The anchor is repl2-eager-dcp1 (submit-repl.sh:25 in
# 2026-09-02-dflash-dcp-port), which ran exactly this shape -- DFlash2,
# DCP=1, SEQS=1, run-server.sh's default 400K -- and reported:
#
#     5.6265 AL, 116.74 tok/s over 64 GSM8K samples
#     Available KV cache memory: 22.28 GiB
#     GPU KV cache size: 400,064 tokens
#     2317 hot / 2483 cold experts per rank (45.8 GiB available)
#
# That arm served NVFP4, not W4A16. KV sizing is independent of the weight
# quantisation -- the planner derives num_blocks from max_model_len and
# max_num_seqs alone (tiered_moe_kv.py:309-312) -- so the cache figures carry
# over unchanged. Expert residency does not: W4A16 weights are smaller, so
# expect more hot experts than 2317. The residency log line reports it.
#
# Why 400,064 tokens is the whole context and not a budget: at c=1 the
# planner asks for ceil(400000/64) + 1 = 6251 blocks, one full-length
# sequence. The c=4 launcher's problem is that this term scales with
# max_num_seqs while DCP 1 leaves the MLA cache replicated rather than
# sharded, so four sequences want four times the per-rank bytes. At c=1 that
# pressure is gone and the context is free.
#
# Deliberately NO --kv-cache-memory, unlike every other W4A16 config here
# (they all pass 21689598771, which buys 400,064 tokens per rank under MTP3).
# That flag only short-circuits the profiling run (gpu_worker.py:471); it is
# not an input to the tiered planner. DFlash2 charges six dense draft layers
# of KV per block where MTP3 charges one grafted MLA layer, so the MTP3
# constant is wrong here, and deriving a replacement is what cost the c=4
# launcher three revisions. repl2-eager-dcp1 passed no flag and profiled to a
# self-consistent 22.28 GiB. The cost is ~3.5 min of startup profiling, which
# the 240 x 10s ready loop below absorbs.
#
# Confirmed by launch: jobs 1658089 and 1658174 both profiled to 21.92 GiB and
# allocated exactly 400,064 tokens, the planner's 6251 blocks. They died later,
# on HBM headroom rather than KV sizing -- see the reserve note below.
#
# Differences from claude-glm53-c4.sh (MTP3, DCP4), all deliberate:
#   * speculator  DFlash2 (GLM-5.3-DFlash2), 7 speculative tokens, greedy
#                 draft sampling, FLASH_ATTN draft attention. Verifies 8
#                 tokens per step against MTP3's 4.
#   * dcp         1, not 4. See above.
#   * seqs        1, not 4.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/claude-glm53-c1-df2}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-09-04-glm53-c1-df2}"
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
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-glm53-c1-df2
export TRTLLM_DG_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/trtllm-dg-glm53-c1-df2
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
# 10 GB reserve at utilisation 0.90. The reserve is doing a job the planner
# should do itself, so the number is not arbitrary.
#
# The tiered planner budgets HBM from fixed_hbm_allocations, which holds
# main_mla_cache (tiered_moe_physical.py:227) and indexer_cache (:222) and
# nothing else. The KV cache it then allocates is sized by
# get_tiered_kv_available_memory (tiered_moe_kv.py:140, via
# kv_cache_utils.py:2201), which sums main *and indexer and draft* specs. For
# this shape, per rank at 6251 blocks:
#
#     main_mla_cache   19.06 GiB   budgeted
#     indexer_cache     1.03 GiB   budgeted
#     draft cache       2.18 GiB   NOT budgeted
#
# The drafter's six dense layers are invisible to the planner, which spends
# that 2.18 GiB on hot experts instead and then cannot fit the cache it just
# promised. MTP3 never trips this: its grafted layer is an MLA layer counted
# in main_specs, so main_cache_bytes already covers it. That is why the MTP3
# launcher runs happily at reserve 6 and this one does not.
#
# Reserve is the only lever that moves real headroom, and it moves it 1:1 --
# measured, 6 GB -> 7 GB took experts 50.0 -> 49.1 GiB and free 0.88 -> 1.80
# GiB. Utilisation does NOT: 0.94 -> 0.90 changed only the *declared* KV budget
# (21.92 -> 19.06 GiB), which is slack the tiered path ignores because the
# planner forces the block count either way. Both runs allocated the identical
# 400,064 tokens. Keep 0.90 anyway: it is the precondition for a reserve above
# 7, which at 0.94 fails the reserve check and refuses to start.
#
# So the reserve carries the unbudgeted 2.18 GiB plus margin. Jobs 1658089 and
# 1658174 died at 0.94/6 (0.88 GiB free) and 1658456 at 0.90/7 (1.80 GiB free),
# each short of a 2.00 GiB allocation during KV cache init.
#
# The real fix is to add the draft specs to fixed_hbm_allocations; until then
# every dense-drafter shape needs a reserve inflated by its draft cache.
#
export TIERED_MOE_HBM_RESERVE_GB="${CLAUDE_GLM53_HBM_RESERVE_GB:-10}"
gpu_util="${CLAUDE_GLM53_GPU_UTIL:-0.90}"
# The unbudgeted draft cache scales with the block count, so shortening the
# context shrinks the gap the runtime reserve check trips on. Overridable to
# probe that; 400000 is the shape pin, and anything else needs the pin lifted.
model_len="${CLAUDE_GLM53_MAX_MODEL_LEN:-400000}"
if [[ "${model_len}" != 400000 ]]; then
  export VLLM_TIERED_MOE_RELAX_SHAPE=1
fi
echo "memory: gpu_util=${gpu_util} reserve=${TIERED_MOE_HBM_RESERVE_GB}GB max_model_len=${model_len}"
# Stage each layer's cold experts into an HBM slot while the previous layer
# computes, so the cold tier reads HBM instead of scattered Grace. On this
# checkpoint (GLM-5.3 W4A16): cold Marlin 580 -> 224 ms/chunk at 333/351 GB/s
# against the hot tier's 335/353, prefill -16.1% (23.36 -> 19.60 s on the 96K
# prompt, three reps each, both arms in one allocation). GSM8K unchanged:
# paired McNemar p=0.888 on the fork branch, p=1.000 on the single-stream
# path. See agent_space/experiments/2026-09-05-cold-prefetch/PHASE5.md.
#
# The gate is a token count, not a phase test: it must exceed the largest
# decode batch, which setup_tiered_moe_kernels enforces at load. Here that is
# max_num_seqs x verify tokens = 1 x 8 = 8, so 1024 clears it with room.
#
# Interaction with the reserve above: the planner budgets 871 MiB for the slot
# and pays for it by demoting 43 hot experts (2192 -> 2149), so this is
# residency the plan gives up, not reserve the drafter needs.
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="${CLAUDE_GLM53_COLD_PREFETCH_MIN_TOKENS:-1024}"
# The drafter runs eagerly by default now: DFlash2Speculator sets
# default_eager_draft, because replaying the captured draft graph costs
# acceptance (Phase 52, undiagnosed). On GSM8K at DCP1, same config, the mode
# being the only difference: 5.7675 AL / 117.6 tok/s eager against 3.9699 /
# 90.8 captured. This launcher ran captured until then and measured 2.494 AL on
# live traffic. Set CLAUDE_GLM53_EAGER_DRAFT=0 to measure the captured path.
if [[ -n "${CLAUDE_GLM53_EAGER_DRAFT:-}" ]]; then
  export VLLM_DFLASH2_EAGER_DRAFT="${CLAUDE_GLM53_EAGER_DRAFT}"
fi
echo "draft: eager=${VLLM_DFLASH2_EAGER_DRAFT:-default (DFlash2 -> eager)}"
# DFlash2 verifies 8 tokens per step (7 drafted plus the bonus), and c=1
# means that is the only decode shape: capture [8] and nothing else, matching
# arm-dcp.sh:34,46 at SEQS=1.
#
# compile_sizes is EMPTY, unlike the MTP3 launcher's [4,8,12,16]. Every working
# DFlash2 arm runs it empty (2026-09-02-dflash-dcp-port/arm-dcp.sh:34), for two
# measured reasons. Inductor autotuning at these shapes asks for more shared
# memory than an SM has -- the c=4 sibling's first revision set
# compile_sizes=[8,16,24,32] and died at startup with
#   No valid triton configs. OutOfMemoryError: out of resource: triton_mm
#   Required: 262144  Hardware limit: 232448
# -- and separately a non-empty compile_sizes makes the DFlash2 selector a
# second compiled unit inside the draft's captured region, which aborts with
# "Expected exactly one compiled range_entry ... found 2" (see
# VLLM_DFLASH2_NO_SELECTOR_COMPILE).
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

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
  --max-num-seqs 1 \
  --host 0.0.0.0 \
  --port "${port}" \
  --gpu-memory-utilization "${gpu_util}" \
  --max-model-len "${model_len}" \
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
