#!/usr/bin/env bash
# One DFlash2 acceptance arm. Runs ON the compute node, inside an existing
# allocation, launched by run-in-alloc.sh.
#
# No MTP3 control arm. The baseline is settled: three replicates on three
# nodes gave 3.0783, 3.0659 and 3.0599 acceptance at 27.52, 27.86 and 27.37
# ms, a 0.6% spread. Re-measuring it on every bring-up iteration is waste.
# Use job-c1-acceptance.sh if a fresh matched control is ever needed again.

set -euo pipefail

label="${1:?result label is required}"
spec_tokens="${2:-7}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-dflash2-backport"
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
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-dflash2"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-dflash2"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
[[ -f "${model}/.stage_done" ]] || model="${models_dir}/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
echo "model:  ${model}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${result_dir}/dflash2-trim-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=7

export DFLASH_DRAFT_PATH="${DFLASH_DRAFT_PATH:-${models_dir}/GLM-5.3-DFlash2}"
echo "draft:  ${DFLASH_DRAFT_PATH}"
"${result_dir}/run-server-dflash2.sh" "${spec_tokens}" 1 "" \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

ready=false
for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "(ValueError|RuntimeError|AssertionError|ValidationError|NotImplementedError|KeyError|IndexError|TypeError|AttributeError): .*" \
      "${result_dir}/${label}-server.out" "${result_dir}/${label}-server.err" 2>/dev/null | sort -u | head -5
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" != true ]]; then
  echo "SERVER NOT READY"; kill "${pid}" 2>/dev/null || true; exit 1
fi

curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":13}' \
  -o "${result_dir}/${label}-semantic.json"
echo "--- semantic: $(jq -r '.choices[0].text' "${result_dir}/${label}-semantic.json")"
echo "--- MTP3 on this prompt gives: ' Paris. Distance from Paris to Lyon is'"

rc=0
.venv/bin/python "${result_dir}/capture_acceptance.py" \
  --label "${label}" --out "${result_dir}/acceptance-${label}.json" || rc=$?

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
exit "${rc}"
