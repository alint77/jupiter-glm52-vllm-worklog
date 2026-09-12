#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=decode-graph-acc
#SBATCH --output=agent_space/experiments/2026-09-05-decode-placement-replicas/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-decode-placement-replicas/slurm-%j.err
set -euo pipefail
ulimit -c 0
src=/e/fscratch/profound/naeimitabiei1/worktrees/decode-placement-replicas-qualification
cd "${src}"
source agent_space/jupiter-env.sh
export PYTHONPATH="${src}"
export PYTHONFAULTHANDLER=1
here="${src}/agent_space/experiments/2026-09-05-decode-placement-replicas"
out="${here}/results-${SLURM_JOB_ID}"
mkdir -p "${out}"
cp "${here}/qualification-provenance.json" "${out}/provenance.json"
hostname >"${out}/node.txt"
cache="/e/fscratch/profound/${USER}/caches/decode-placement-dflash2"
export VLLM_CACHE_ROOT="${cache}/vllm" TRTLLM_DG_CACHE_DIR="${cache}/dg"
export TRITON_CACHE_DIR="${cache}/triton" TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
export FLASHINFER_CACHE_DIR="${cache}/flashinfer"
export TMPDIR="/e/fscratch/profound/${USER}/caches/gsm8k"
export PREFETCH_MIN_TOKENS=1024
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"

baseline="${here}/baseline-profile.json"
candidate="${here}/df2-prefetch-cap50-joint/runtime.json"

pid=""
cleanup() {
  if [[ -n "${pid}" ]]; then
    kill -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    pid=""
  fi
}
trap cleanup EXIT INT TERM

cap8='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
cap16='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[16],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

eager='{"mode":0,"cudagraph_mode":"NONE","compile_sizes":[],"pass_config":{"fuse_allreduce_rms":false}}'

# arm <tag> <profile> <assignment> <route_check> <compilation_config> <trace>
arm() {
  local tag="$1" profile="$2" assignment="$3" check="$4" cfg="$5" trace="${6:-0}"
  echo "=== arm ${tag}: assignment=${assignment} CHECK=${check} TRACE=${trace} ==="
  ROUTE_CHECK="${check}" TIERED_MOE_COMPILATION_CONFIG="${cfg}" \
    VLLM_TIERED_MOE_ROUTE_TRACE="${trace}" \
    setsid bash "${here}/serve-dflash2.sh" \
    "${profile}" "${assignment}" \
    >"${out}/${tag}-server.out" 2>"${out}/${tag}-server.err" &
  pid=$!
  local ready=false
  for _ in $(seq 1 480); do
    if curl -fsS http://127.0.0.1:8129/health >/dev/null 2>&1; then
      ready=true; break
    fi
    kill -0 "${pid}" 2>/dev/null || break
    sleep 5
  done
  if [[ "${ready}" != true ]]; then
    echo "[${tag}] SERVER DID NOT START" | tee "${out}/${tag}-FAILED.txt"
    tail -40 "${out}/${tag}-server.err" >>"${out}/${tag}-FAILED.txt" || true
    cleanup; sleep 20; return 0
  fi
  .venv/bin/python "${here}/diagnose_smoke.py" --arm "${tag}" \
    --num-questions 24 --num-shots 5 --max-tokens 256 --port 8129 \
    --out "${out}/${tag}-smoke.json" || echo "[${tag}] smoke script error"
  cleanup
  sleep 20
}

# The tracer found no divergence at all in eager mode over 24 questions, while
# the fatal check under graphs keeps asserting. Python does not execute during
# graph replay, so the tracer cannot see that path. Ask the only question that
# decides whether it matters: with graphs on and the check disabled, is the
# output actually wrong? Healthy accuracy means the routes are effectively
# fine and the in-graph check is reporting a divergence that does not exist.
arm graph-nocheck "${candidate}" exact 0 "${cap8}" 0
arm eager-control "${candidate}" exact 0 "${eager}" 0

echo "=== ladder ==="
for f in "${out}"/*-smoke.json; do
  [[ -e "${f}" ]] || continue
  .venv/bin/python -c "
import json,sys
d=json.load(open('${f}'))
print('%-16s accuracy %.3f  invalid %.3f  preds %s' % (
    d['arm'], d['accuracy'], d['invalid'], d['preds'][:6]))
"
done

