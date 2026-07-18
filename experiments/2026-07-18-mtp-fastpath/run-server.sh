#!/usr/bin/env bash

set -euo pipefail

depth="${1:?speculative depth is required}"
local_argmax="${2:?local-argmax setting is required}"
profile_dir="${3:-}"
shift 3

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
verification_size=$((depth + 1))
speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${depth},\"use_local_argmax_reduction\":${local_argmax}}"

export TIERED_MOE_MODEL_PATH="$(dirname -- "${repo_dir}")/models/GLM-5.2-W4A16-FP8-MTP"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-18-mtp-graft/placement-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=10
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${verification_size}],\"compile_sizes\":[${verification_size}],\"pass_config\":{\"fuse_allreduce_rms\":false}}"

profiler_args=()
if [[ -n "${profile_dir}" ]]; then
  mkdir -p "${profile_dir}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":50,\"max_iterations\":8}"
  profiler_args=(--profiler-config "${profiler_config}")
fi

cd "${repo_dir}"
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${speculative_config}" \
  "${profiler_args[@]}" \
  "$@"
