#!/usr/bin/env bash
# Phase 3b: does the staging path move prefill output, and is the baseline
# even reproducible? Four server loads in one allocation. The gate is a
# logprob distance, not token equality -- a greedy argmax can flip on a tie.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:30:00
#SBATCH --job-name=cpfphase3b
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
    >"${result_dir}/p3b-${arm}-server.out" \
    2>"${result_dir}/p3b-${arm}-server.err" &
  local pid=$!

  local ready=false
  for _ in $(seq 1 400); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SERVER EXITED EARLY (${arm})"
      grep -ohE "CUDA out of memory[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
        "${result_dir}/p3b-${arm}-server."{out,err} 2>/dev/null \
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

full = json.loads(Path(
    "agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl"
).read_text().splitlines()[0])["prompt"]
# Deterministic slices: one chunk, and the 1024-2048 overlap window.
prompts = {"long": full, "chunk": full[:16000], "mid": full[:6000]}


def post(prompt, max_tokens, logprobs=None):
    body = {"model": "glm53-cmp-tiered", "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0}
    if logprobs is not None:
        body["logprobs"] = logprobs
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.loads(r.read().decode())
    return payload, time.perf_counter() - start


records = []
for name, prompt in prompts.items():
    for rep in range(3):
        # The gate: the prefill's own output distribution, one token deep.
        probe, seconds = post(prompt, 1, logprobs=20)
        choice = probe["choices"][0]
        top = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
        text, text_seconds = post(prompt, 32)
        records.append({
            "arm": arm, "prompt": name, "rep": rep,
            "prompt_tokens": probe.get("usage", {}).get("prompt_tokens"),
            "top_logprobs": top[0],
            "first_token": choice["text"],
            "text": text["choices"][0]["text"],
            "probe_seconds": seconds, "text_seconds": text_seconds,
        })
        print(f"{arm} {name} rep{rep}: {seconds:.2f}s/{text_seconds:.2f}s "
              f"first={choice['text']!r}")

(out / f"p3b-{arm}.json").write_text(json.dumps(records, indent=2))
PY

  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 20
}

# Two identical baseline arms: the second is the across-restart control.
run_arm baseline1 0 0
run_arm baseline2 0 0
# Staging on, slot read. VERIFY off separates the per-layer host sync from
# the slot itself; VERIFY on reproduces the phase-3 configuration exactly.
run_arm staged_noverify 1024 0
run_arm staged_verify 1024 1

echo "=== staging health ==="
grep -hoE "cold prefetch: chunk .{0,160}" \
  "${result_dir}/p3b-staged_verify-server."{out,err} 2>/dev/null | tail -3
echo "=== done ==="
