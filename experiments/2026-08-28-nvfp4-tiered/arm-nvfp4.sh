#!/usr/bin/env bash
# NVFP4 bring-up under the tiered contract. Runs ON the compute node inside an
# existing allocation, launched by run-in-alloc.sh from the sibling phase.
#
# No speculator: MTP3 would instantiate layer 78, whose experts are BF16 in
# this checkpoint and cost 4.50 GiB/rank rather than W4G64's 1.21. Get the
# target serving first, then add DFlash2, which does not instantiate layer 78
# at all.
#
# The placement profile carries the GLM-5.2 ranking and is a bring-up
# placeholder. Placement affects neither loading nor output, so it is fine
# here and must not be used for any performance number.

set -euo pipefail

label="${1:?result label is required}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
models_dir=/e/project1/profound/alint77/models

cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node:   $(hostname)"
head_ref="$(<"${repo_dir}/.git/HEAD")"
if [[ "${head_ref}" == ref:* ]]; then
  branch="${head_ref#ref: refs/heads/}"; commit="$(<"${repo_dir}/.git/refs/heads/${branch}")"
else
  branch="(detached)"; commit="${head_ref}"
fi
echo "commit: ${commit:0:10} (${branch})"
echo "label:  ${label}"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-nvfp4"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-nvfp4"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
[[ -f "${model}/.stage_done" ]] || model="${models_dir}/GLM-5.3-NVFP4"
echo "model:  ${model}"

export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${TIERED_MOE_PLACEMENT_PROFILE:-${result_dir}/nvfp4-placeholder-profile.json}"
export TIERED_MOE_HBM_RESERVE_GB="${TIERED_MOE_HBM_RESERVE_GB:-7}"
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --max-num-seqs 1 \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

ready=false
for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "(ValueError|RuntimeError|AssertionError|ValidationError|NotImplementedError|KeyError|IndexError|TypeError|AttributeError): .*" \
      "${result_dir}/${label}-server.out" "${result_dir}/${label}-server.err" 2>/dev/null | sort -u | head -5
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]] || { echo "SERVER NOT READY"; kill "${pid}" 2>/dev/null || true; exit 1; }

echo "--- MoE backend actually selected ---"
grep -ohE "Using '[^']+' NvFp4 MoE backend|Using '[^']+' WNA16 MoE backend|MARLIN.*MoE backend" \
  "${result_dir}/${label}-server.out" 2>/dev/null | sort -u | head -3

for prompt in "The capital of France is" "Write a Python function to reverse a linked list."; do
  echo "--- prompt: ${prompt}"
  curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg p "${prompt}" '{model:"glm52-w4a16-tiered",prompt:$p,max_tokens:40,temperature:0,seed:13}')" \
    | jq -r '.choices[0].text'
done

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
