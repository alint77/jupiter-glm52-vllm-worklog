#!/usr/bin/env bash
# Phase 3: execute the cold tier from the staged HBM slot.
# Both arms run in one allocation against the same prompt at temperature 0.
# The gate is a byte-identical completion; prefill wall time is the cheap
# read on whether cold Marlin actually got faster.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:30:00
#SBATCH --job-name=cpfphase3
#SBATCH --output=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-09-05-cold-prefetch"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SERVER_DEV_MODE=1
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=10
cache_root="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin"
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-cpf3"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-cpf3"
export TRITON_CACHE_DIR="${cache_root}/triton-cpf3"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-cpf3"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-cpf3"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

echo "node $(hostname)"

run_arm() {
  local arm="$1"
  export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="$2"
  export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY="$3"
  echo "=== arm ${arm}: MIN_TOKENS=$2 VERIFY=$3 ==="

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --decode-context-parallel-size 1 --max-num-seqs 1 \
    --served-model-name glm53-cmp-tiered \
    --gpu-memory-utilization 0.90 --max-model-len 400000 \
    --no-enable-prefix-caching \
    >"${result_dir}/phase3-${arm}-server.out" \
    2>"${result_dir}/phase3-${arm}-server.err" &
  local pid=$!

  local ready=false
  for _ in $(seq 1 400); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SERVER EXITED EARLY (${arm})"
      grep -ohE "CUDA out of memory[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
        "${result_dir}/phase3-${arm}-server."{out,err} 2>/dev/null \
        | grep -viE "Engine core init|WorkerProc init" | sort -u | head -5
      return 1
    fi
    sleep 5
  done
  [[ "${ready}" == true ]] || return 1
  echo "server ready (${arm})"

  ARM="${arm}" RESULT_DIR="${result_dir}" .venv/bin/python - <<'PY'
import json, os, time, urllib.request
from pathlib import Path

arm = os.environ["ARM"]
out = Path(os.environ["RESULT_DIR"])
url = "http://127.0.0.1:8027/v1/completions"


def ask(prompt, max_tokens):
    body = json.dumps({"model": "glm53-cmp-tiered", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.loads(r.read().decode())
    return payload["choices"][0]["text"], time.perf_counter() - start


short, _ = ask("The capital of France is", 8)
(out / f"phase3-{arm}-short.txt").write_text(short)
print(f"{arm} short: {short!r}")

prompt = json.loads(Path(
    "agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl"
).read_text().splitlines()[0])["prompt"]
# One untimed pass so neither arm pays first-touch costs in its number.
ask(prompt, 4)
text, elapsed = ask(prompt, 32)
(out / f"phase3-{arm}-long.txt").write_text(text)
(out / f"phase3-{arm}-timing.json").write_text(
    json.dumps({"arm": arm, "seconds": elapsed}, indent=2)
)
print(f"{arm} long: {elapsed:.2f}s")
PY

  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 20
}

run_arm baseline 0 0
run_arm prefetch 1024 1

echo "=== prefetch log lines ==="
grep -hoE "cold prefetch: .{0,180}" \
  "${result_dir}/phase3-prefetch-server."{out,err} 2>/dev/null | tail -20
echo "=== mismatches (must be none) ==="
grep -hc "do not match its Grace tier" \
  "${result_dir}/phase3-prefetch-server."{out,err} 2>/dev/null || true

echo "=== prefill wall time ==="
cat "${result_dir}/phase3-"{baseline,prefetch}"-timing.json" || true

gate=0
for kind in short long; do
  echo "=== ${kind} completion diff ==="
  if diff "${result_dir}/phase3-baseline-${kind}.txt" \
          "${result_dir}/phase3-prefetch-${kind}.txt"; then
    echo "IDENTICAL (${kind})"
  else
    echo "DIFFERS (${kind}) -- gate failed"
    gate=1
  fi
done
echo "=== done, gate=${gate} ==="
exit "${gate}"
