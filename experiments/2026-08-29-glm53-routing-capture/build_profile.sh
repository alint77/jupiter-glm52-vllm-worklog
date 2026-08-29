#!/usr/bin/env bash
# Turn a routing-capture trace directory into a GLM-5.3 placement profile.
#
#   ./build_profile.sh <trace-dir> <out-prefix> [hot-slots-per-rank]
#
# Emits <out-prefix>-profile.json and <out-prefix>-report.json. The profile is
# schema version 1 (no replicas); add_secondary_ranks.py promotes it to the
# version 2 layout the shipped nvfp4-profile-2400.json uses.

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
trace_dir="${1:?usage: build_profile.sh <trace-dir> <out-prefix> [slots]}"
prefix="${2:?usage: build_profile.sh <trace-dir> <out-prefix> [slots]}"
slots="${3:-2400}"
model="${TIERED_MOE_MODEL_PATH:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4}"

"${repo_dir}/.venv/bin/python" \
  "$(dirname -- "${BASH_SOURCE[0]}")/traces_to_manifest.py" \
  --trace-dir "${trace_dir}" \
  ${ATTRIBUTION:+--attribution "${ATTRIBUTION}"}

for mode in frequency tail; do
  "${repo_dir}/.venv/bin/python" \
    "${repo_dir}/agent_space/benchmarks/optimize_routing_profile.py" \
    --trace-dir "${trace_dir}" \
    --model "${model}" \
    --hot-slots-per-rank "${slots}" \
    --residency-mode "${mode}" \
    --output-profile "${prefix}-${mode}-profile.json" \
    --output-report "${prefix}-${mode}-report.json" \
    >"${prefix}-${mode}-stdout.json"
  printf 'wrote %s-%s-profile.json\n' "${prefix}" "${mode}"
done
