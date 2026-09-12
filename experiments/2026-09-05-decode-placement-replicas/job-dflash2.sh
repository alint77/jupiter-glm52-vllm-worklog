#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=decode-df2-joint
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
cp "${here}/baseline-profile.json" "${out}/baseline-profile.json"
cp "${CANDIDATE_PROFILE:-${here}/df2-prefetch-cap50-joint/runtime.json}" "${out}/candidate-profile.json"
hostname >"${out}/node.txt"
cache="/e/fscratch/profound/${USER}/caches/decode-placement-dflash2"
export VLLM_CACHE_ROOT="${cache}/vllm" TRTLLM_DG_CACHE_DIR="${cache}/dg"
export TRITON_CACHE_DIR="${cache}/triton" TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
export FLASHINFER_CACHE_DIR="${cache}/flashinfer"
export TMPDIR="/e/fscratch/profound/${USER}/caches/gsm8k"
export PREFETCH_MIN_TOKENS=1024
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
.venv/bin/python -m pytest tests/model_executor/model_loader/test_tiered_moe_replica.py \
  -q >"${out}/replica-tests.txt" 2>&1
pid=""
cleanup() {
  if [[ -n "${pid}" ]]; then
    kill -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    pid=""
  fi
}
trap cleanup EXIT INT TERM
start() {
  local tag="$1" profile="$2" assignment="$3"
  setsid bash "${here}/serve-dflash2.sh" "${profile}" "${assignment}" \
    >"${out}/${tag}-server.out" 2>"${out}/${tag}-server.err" &
  pid=$!
  local ready=false
  for _ in $(seq 1 480); do
    if curl -fsS http://127.0.0.1:8129/health >/dev/null 2>&1; then
      ready=true; break
    fi
    kill -0 "${pid}" 2>/dev/null || { tail -70 "${out}/${tag}-server.err"; exit 1; }
    sleep 5
  done
  [[ "${ready}" == true ]]
}
eval_model() {
  local tag="$1"
  .venv/bin/python agent_space/experiments/2026-09-05-cold-prefetch/gsm8k_paired.py \
    --num-questions 1319 --num-shots 5 --max-tokens 256 --port 8129 \
    --out "${out}/${tag}-gsm8k.json"
}
export ROUTE_CHECK=1
start route-smoke "${out}/candidate-profile.json" exact
.venv/bin/python agent_space/experiments/2026-09-05-cold-prefetch/gsm8k_paired.py \
  --num-questions 8 --num-shots 5 --max-tokens 256 --port 8129 \
  --out "${out}/route-smoke-gsm8k.json"
# gsm8k_paired only refuses `invalid > 0.5`, so a server that died partway
# through scored exactly 0.5 and passed twice. Any unanswered request here is
# a dead server, not a wrong answer: refuse the run outright.
.venv/bin/python - "${out}/route-smoke-gsm8k.json" <<'GATE'
import json, sys

sys.path.insert(0, "agent_space/experiments/2026-09-05-cold-prefetch")
from gsm8k_paired import INVALID

r = json.load(open(sys.argv[1]))
unanswered = sum(p == INVALID for p in r["preds"])
if unanswered:
    sys.exit(f"route smoke: {unanswered} unanswered; the server did not survive")
if r["accuracy"] < 0.5:
    sys.exit(f"route smoke accuracy {r['accuracy']:.3f} is below 0.5")
print(f"route smoke ok: accuracy {r['accuracy']:.3f}, all requests answered")
GATE
cleanup
sleep 15
for round in 1 2 3; do
  arms=(baseline candidate)
  [[ "${round}" == 2 ]] && arms=(candidate baseline)
  for arm in "${arms[@]}"; do
    tag="r${round}-${arm}"
    assignment=off
    [[ "${arm}" == candidate ]] && assignment=exact
    export ROUTE_CHECK=0
    start "${tag}" "${out}/${arm}-profile.json" "${assignment}"
    .venv/bin/python "${here}/measure.py" --out "${out}/${tag}.json" \
      --prompts agent_space/experiments/2026-07-29-marlin-smem-monopoly/agentic-prompts.jsonl \
      --seqs 1 --count 12 --tokens 256
    .venv/bin/python "${here}/measure.py" --out "${out}/${tag}-long.json" \
      --prompts agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl \
      --seqs 1 --count 3 --tokens 128
    if [[ "${round}" == 1 && "${arm}" == baseline ]]; then
      eval_model baseline
    fi
    cleanup
    sleep 15
  done
done
export ROUTE_CHECK=1
start route-check "${out}/candidate-profile.json" exact
eval_model candidate
cleanup

echo "=== paired summary and noninferiority gate ==="
.venv/bin/python "${here}/summarize.py" "${out}"
