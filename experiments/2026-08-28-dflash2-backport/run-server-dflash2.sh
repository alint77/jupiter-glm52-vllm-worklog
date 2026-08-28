#!/usr/bin/env bash
# DFlash2 launcher. Carries forward every operational lesson from the Phase 28
# DFlash1 bring-up, which are all still live:
#
#   - The tiered planner budgets draft weights only when method == "mtp"
#     (tiered_moe_physical.py), so the 4.58 GiB bf16 DFlash2 draft is invisible
#     to it. Raising TIERED_MOE_HBM_RESERVE_GB does not substitute: the
#     fail-closed audit's required_free scales 1:1 with the planned reserve.
#     The profile must be trimmed instead.
#   - The placement profile carries a config_sha256 fingerprint of the target,
#     and the tiered loader fails closed on a mismatch, so the target and the
#     profile must be a matched pair.
#   - The draft is a dense qwen3 backbone, not MLA, so it cannot inherit the
#     target's fp8_ds_mla KV dtype; speculative_config carries its own.
#
# New for DFlash2:
#   - block_size 8, so verification batch is spec_tokens + 1 = 8 at c1.
#   - layer_types are all sliding_attention and dflash_config sets
#     `is_causal: false` at the top level, so the draft is non-causal and needs
#     a non-causal-capable backend, as DFlash1 did.
#   - _is_dflash2_draft() forces the V2 model runner unconditionally
#     (config/vllm.py:560). No V1 fallback remains reachable for this draft.

set -euo pipefail

spec_tokens="${1:?num_speculative_tokens is required (block_size 8 -> 7)}"
max_num_seqs="${2:?max concurrent sequences is required}"
profile_dir="${3:-}"
shift 3

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
models_dir="$(dirname -- "${repo_dir}")/models"
draft="${models_dir}/GLM-5.3-DFlash2"

[[ -d "${draft}" ]] || { echo "missing draft: ${draft}" >&2; exit 1; }

# DFlash drafts a block of spec_tokens; verification batch is spec_tokens + 1.
verification_size=$((spec_tokens + 1))
sizes=""
for ((i = 1; i <= max_num_seqs; i++)); do
  sizes+="${sizes:+,}$((verification_size * i))"
done

draft_kv_dtype="${DFLASH2_DRAFT_KV_DTYPE:-auto}"
draft_backend="${DFLASH2_DRAFT_BACKEND:-FLASH_ATTN}"
speculative_config="{\"method\":\"dflash\",\"model\":\"${draft}\",\"num_speculative_tokens\":${spec_tokens},\"kv_cache_dtype\":\"${draft_kv_dtype}\",\"attention_backend\":\"${draft_backend}\"}"

# Target and profile must be a fingerprinted pair. Both are overridable so the
# same launcher serves the acceptance arm (trimmed profile, residency
# irrelevant) and any later throughput arm.
export TIERED_MOE_MODEL_PATH="${TIERED_MOE_MODEL_PATH:-${models_dir}/GLM-5.2-AutoRound-W4G64-MTP-e1ba887}"
export TIERED_MOE_PLACEMENT_PROFILE="${TIERED_MOE_PLACEMENT_PROFILE:?a draft-aware trimmed profile is required; see README}"
export VLLM_TIERED_MOE_PROFILE_CAP="${VLLM_TIERED_MOE_PROFILE_CAP:-1}"
export TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-7}"
# compile_sizes is deliberately EMPTY here, unlike every other arm.
#
# DFlash2's CandidateSelector takes only statically shaped inputs, so
# piecewise_backend takes its "all inputs have static shapes" branch, which
# asserts exactly one compiled range entry. A non-empty compile_sizes gives it
# two -- the static entry for that size, plus the range entry the engine
# derives automatically to compile_ranges_endpoints -- and the selector dies
# with "Expected exactly one compiled range_entry for static shape
# compilation, but found 2" (job 1525715). MTP3 never hits this because it has
# no static-shape-only submodule.
#
# cudagraph_capture_sizes still carries the width-8 shape, so the graph is
# captured as before; only the extra static compile entry is dropped.
compile_sizes="${DFLASH2_COMPILE_SIZES-}"
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${sizes}],\"compile_sizes\":[${compile_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

profiler_args=()
if [[ -n "${profile_dir}" ]]; then
  mkdir -p "${profile_dir}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":${VLLM_TORCH_PROFILER_DELAY_ITERATIONS:-50},\"max_iterations\":8}"
  profiler_args=(--profiler-config "${profiler_config}")
fi

cd "${repo_dir}"
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${speculative_config}" \
  --max-num-seqs "${max_num_seqs}" \
  "${profiler_args[@]}" \
  "$@"
