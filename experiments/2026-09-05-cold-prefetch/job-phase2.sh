#!/usr/bin/env bash
# Phase 2 in situ: stage every layer, verify the bytes, execute from Grace.
# Nothing reads the slot, so any change in output would itself be a bug.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=cpfphase2
#SBATCH --output=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-09-05-cold-prefetch"
profile_dir="${repo_dir}/agent_space/experiments/2026-09-04-mtp3-profile"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=10
cache_root="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin"
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-cpf2"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-cpf2"
export TRITON_CACHE_DIR="${cache_root}/triton-cpf2"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-cpf2"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-cpf2"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

# The feature under test. A prefill chunk is 8192 tokens; 1024 keeps decode
# (4 tokens/step) and any short prompt on the existing path.
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024
export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=1

echo "node $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "${result_dir}/mem-before-${SLURM_JOB_ID}.csv"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 1 --max-num-seqs 1 \
  --served-model-name glm53-cmp-tiered \
  --gpu-memory-utilization 0.90 --max-model-len 400000 \
  --no-enable-prefix-caching \
  >"${result_dir}/phase2-server.out" 2>"${result_dir}/phase2-server.err" &
pid=$!
trap 'kill "${pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 400); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "CUDA out of memory[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
      "${result_dir}/phase2-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -5
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]]
echo "server ready"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "${result_dir}/mem-after-load-${SLURM_JOB_ID}.csv"

# Semantics must be untouched: nothing reads the slot in phase 2.
curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-cmp-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | tee "${result_dir}/phase2-semantic-${SLURM_JOB_ID}.json"
echo

# One ~96K prompt drives ~12 prefill chunks, so ~900 staged layers.
.venv/bin/python - <<'PY' > "${result_dir}/phase2-prefill-${SLURM_JOB_ID}.json"
import json, urllib.request
from pathlib import Path
prompt = json.loads(Path("agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl").read_text().splitlines()[0])["prompt"]
body = json.dumps({"model": "glm53-cmp-tiered", "prompt": prompt,
                   "max_tokens": 32, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8027/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    print(r.read().decode())
PY

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "${result_dir}/mem-after-prefill-${SLURM_JOB_ID}.csv"
echo "=== prefetch log lines ==="
grep -hoE "cold prefetch: .{0,140}" "${result_dir}/phase2-server."{out,err} 2>/dev/null | tail -20
echo "=== mismatches (must be none) ==="
grep -c "do not match its Grace tier" "${result_dir}/phase2-server."{out,err} 2>/dev/null || true
echo "=== done ==="
