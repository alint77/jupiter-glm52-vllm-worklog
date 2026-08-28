#!/usr/bin/env bash
# Phase 42 acceptance arm: DFlash2 against a matched MTP3 control, same node,
# same binary, same trimmed placement profile, c1.
#
# Acceptance length is independent of expert residency, so both arms run on the
# trimmed profile and the comparison stays matched. Throughput numbers from
# this job are NOT production-comparable -- residency is 200 hot slots/rank
# below production on both arms. Only the acceptance column is the deliverable.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=dflash2-accept
#SBATCH --output=agent_space/experiments/2026-08-28-dflash2-backport/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-28-dflash2-backport/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-dflash2-backport"
models_dir=/e/project1/profound/alint77/models

cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node:   $(hostname)"
echo "commit: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
git diff --quiet || { echo "REFUSING: working tree is dirty"; exit 1; }

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-dflash2"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-dflash2"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

export TIERED_MOE_MODEL_PATH="${models_dir}/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
export TIERED_MOE_PLACEMENT_PROFILE="${result_dir}/dflash2-trim-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=7

wait_ready() {
  local label="$1" pid="$2"
  for _ in $(seq 1 480); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SERVER EXITED EARLY (${label})"
      tail -40 "${result_dir}/${label}-server.err"
      return 1
    fi
    sleep 10
  done
  echo "SERVER NOT READY (${label})"
  tail -40 "${result_dir}/${label}-server.err"
  return 1
}

smoke() {
  local label="$1"
  curl -fsS http://127.0.0.1:8027/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":13}' \
    -o "${result_dir}/${label}-semantic.json"
  echo "--- ${label} semantic: $(jq -r '.choices[0].text' "${result_dir}/${label}-semantic.json")"
  echo "--- expected a coherent continuation; degenerate output means stop and diagnose"
}

run_arm() {
  local label="$1"; shift
  echo "=============== ${label} ==============="
  "$@" >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
  local pid=$!
  local rc=0
  if wait_ready "${label}" "${pid}"; then
    smoke "${label}"
    .venv/bin/python "${result_dir}/capture_acceptance.py" \
      --label "${label}" \
      --out "${result_dir}/acceptance-${label}-${SLURM_JOB_ID}.json" || rc=$?
  else
    rc=1
  fi
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 20
  return "${rc}"
}

# Control first: if MTP3 does not reproduce its known ~2.9/4 here, the harness
# is wrong and the DFlash2 number would be uninterpretable. Verification width
# is 4 at c1, so the graph must be captured at 4, not run-server.sh's default 1.
mtp3_rc=0
TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}' \
  run_arm mtp3-control \
  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-num-seqs 1 || mtp3_rc=$?
[[ "${mtp3_rc}" -eq 0 ]] || echo "WARNING: control arm failed (rc=${mtp3_rc}); the DFlash2 number below has no matched baseline"

# run-server-dflash2.sh exports its own compilation config sized to width 8.
dflash2_rc=0
run_arm dflash2 "${result_dir}/run-server-dflash2.sh" 7 1 "" || dflash2_rc=$?

echo "=============== summary ==============="
for f in "${result_dir}"/acceptance-*-"${SLURM_JOB_ID}".json; do
  [[ -e "$f" ]] || continue
  jq -c '{label,acceptance_length,draft_acceptance_rate,verification_steps,step_time_ms}' "$f"
done
