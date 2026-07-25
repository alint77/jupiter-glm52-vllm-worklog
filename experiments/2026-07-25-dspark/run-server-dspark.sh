#!/usr/bin/env bash
# DSpark speculator smoke launcher. max_model_len stays at 400000: the tiered
# contract (VllmConfig.validate_tiered_moe) pins it, and it sizes the KV cache
# rather than the prompt, so short smoke prompts are unaffected. Generous
# tiered reserve. The tiered planner only budgets draft weights when
# method == "mtp" (tiered_moe_physical.py), so the ~6.3 GB bf16 DSpark draft is
# invisible to it and must be covered by the HBM reserve by hand.

set -euo pipefail

spec_tokens="${1:?num_speculative_tokens is required (>= block_size 8)}"
max_num_seqs="${2:?max concurrent sequences is required}"
profile_dir="${3:-}"
shift 3

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
models_dir="$(dirname -- "${repo_dir}")/models"
draft="${models_dir}/GLM-5.2-speculator-dspark"

# DSpark drafts a block of `spec_tokens`; verification batch is spec_tokens + 1.
verification_size=$((spec_tokens + 1))
sizes=""
for ((i = 1; i <= max_num_seqs; i++)); do
  sizes+="${sizes:+,}$((verification_size * i))"
done

speculative_config="{\"method\":\"dspark\",\"model\":\"${draft}\",\"num_speculative_tokens\":${spec_tokens}}"

# Must stay the MTP-grafted checkpoint: the placement profile carries a
# config_sha256 fingerprint of it (1c6c98...), and the tiered loader fails
# closed on a mismatch. The grafted MTP layer 78 is simply not instantiated
# under DSpark, so it costs nothing but the fingerprint has to match.
export TIERED_MOE_MODEL_PATH="${TIERED_MOE_MODEL_PATH:-${models_dir}/GLM-5.2-W4A16-FP8-MTP}"
export TIERED_MOE_PLACEMENT_PROFILE="${TIERED_MOE_PLACEMENT_PROFILE:-${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/per-expert-profile.json}"
# Headroom for the unbudgeted draft weights + its sliding-window KV.
export TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-16}"
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${sizes}],\"compile_sizes\":[${sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

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
