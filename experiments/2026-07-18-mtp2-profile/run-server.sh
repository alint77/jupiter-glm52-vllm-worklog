#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
result_dir="${repo_dir}/agent_space/experiments/2026-07-18-mtp2-profile"
profile_dir="/e/scratch/profound/naeimitabiei1/glm52-mtp2-profile-${SLURM_JOB_ID}"

export TIERED_MOE_MODEL_PATH="$(dirname -- "${repo_dir}")/models/GLM-5.2-W4A16-FP8-MTP"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-18-mtp-graft/placement-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=10
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3],"compile_sizes":[3],"pass_config":{"fuse_allreduce_rms":false}}'

mkdir -p "${profile_dir}"
profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":50,\"max_iterations\":8}"

cd "${repo_dir}"
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --profiler-config "${profiler_config}" \
  "$@"
