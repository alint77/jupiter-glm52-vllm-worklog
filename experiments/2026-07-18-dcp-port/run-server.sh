#!/usr/bin/env bash

set -euo pipefail

depth="${1:?speculative depth is required}"
dcp_size="${2:?decode-context-parallel size is required}"
cudagraph_mode="${3:?cudagraph mode is required}"
profile_dir="${4:-}"
shift 4

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
verification_size=$((depth + 1))
speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${depth}}"

export TIERED_MOE_MODEL_PATH="$(dirname -- "${repo_dir}")/models/GLM-5.2-W4A16-FP8-MTP"
export TIERED_MOE_PLACEMENT_PROFILE="${TIERED_MOE_PLACEMENT_PROFILE:-${repo_dir}/agent_space/experiments/2026-07-18-mtp-graft/placement-profile.json}"
export TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-10}"
# One eager warmup per capture size: DCP's decode-shape Triton merge kernel
# and collective paths must not initialize lazily inside graph capture.
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${cudagraph_mode}\",\"cudagraph_capture_sizes\":[${verification_size}],\"compile_sizes\":[${verification_size}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"

dcp_args=()
if [[ "${dcp_size}" != 1 ]]; then
  dcp_args=(--decode-context-parallel-size "${dcp_size}")
fi

profiler_args=()
if [[ -n "${profile_dir}" ]]; then
  mkdir -p "${profile_dir}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":50,\"max_iterations\":8}"
  profiler_args=(--profiler-config "${profiler_config}")
fi

cd "${repo_dir}"
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${speculative_config}" \
  "${dcp_args[@]}" \
  "${profiler_args[@]}" \
  "$@"
