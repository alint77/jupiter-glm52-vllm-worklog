#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --job-name=decode-route-node
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

# arm <tag> <profile> <assignment> <route_check> <compilation_config>
arm() {
  local tag="$1" profile="$2" assignment="$3" check="$4" cfg="$5"
  echo "=== arm ${tag}: assignment=${assignment} ROUTE_CHECK=${check} ==="
  ROUTE_CHECK="${check}" TIERED_MOE_COMPILATION_CONFIG="${cfg}" \
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

# One configuration, repeated: candidate / exact / ROUTE_CHECK=1 at the default
# capture size. Job 1763631 ran this 24 times on jpbo-018-36 without a single
# route assertion; job 1763763 asserted twice in three requests on jpbo-036-14.
# Pin the node to test whether cross-rank route divergence is node-specific.
arm route-a "${candidate}" exact 1 "${cap8}"
arm route-b "${candidate}" exact 1 "${cap8}"
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

