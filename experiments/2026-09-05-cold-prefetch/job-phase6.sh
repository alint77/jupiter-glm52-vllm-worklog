#!/usr/bin/env bash
# Phase 6: the accuracy gate on the single-stream chunked-prefill path.
#
# Neither deferred phase-6 tuning item survived measurement. The launch policy
# is bypassed at M=8192 (marlin_moe.py:142 applies it only when max_tokens >= M,
# and max_tokens is 2048), so it cannot touch the regime staging exists for.
# Wrap-around staging to reach 75/75 is worth 4.82 ms/chunk -- 0.29% of prefill
# wall, below the +/-0.77% measurement noise -- and would let layer 0's bytes
# live in the slot across requests with nothing to invalidate them.
#
# So the allocation goes to the gap instead. Phase 5's GSM8K prompts are ~539
# tokens: one chunk, inside the 2048 overlap window, so it gated the *fork*
# branch. The single-stream M=8192 path -- what the prefetch was actually built
# for -- has byte verification and the roofline behind it but no accuracy eval.
# Padding each prompt to ~9.5K tokens puts chunk 1 at 8192 (single-stream) and
# chunk 2 at ~1349 (staged and forked), covering both branches and the chunk
# boundary in one run.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --job-name=cpfphase6
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
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-cpf6"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-cpf6"
export TRITON_CACHE_DIR="${cache_root}/triton-cpf6"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-cpf6"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-cpf6"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

# GSM8K's loader caches into gettempdir(); the files are pre-staged there so
# the compute node needs no network.
export TMPDIR="/e/fscratch/profound/${USER:-$(id -un)}/caches/gsm8k"
[[ -s "${TMPDIR}/test.jsonl" ]] || { echo "gsm8k not pre-staged" >&2; exit 1; }

echo "node $(hostname)"

run_arm() {
  local arm="$1"
  export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="$2"
  export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=0
  echo "=== arm ${arm}: MIN_TOKENS=$2 ==="

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --decode-context-parallel-size 1 --max-num-seqs 1 \
    --served-model-name glm53-cmp-tiered \
    --gpu-memory-utilization 0.90 --max-model-len 400000 \
    --no-enable-prefix-caching \
    >"${result_dir}/p6-${arm}-server.out" \
    2>"${result_dir}/p6-${arm}-server.err" &
  local pid=$!

  local ready=false
  for _ in $(seq 1 400); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SERVER EXITED EARLY (${arm})"
      grep -ohE "CUDA out of memory[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
        "${result_dir}/p6-${arm}-server."{out,err} 2>/dev/null \
        | grep -viE "Engine core init|WorkerProc init" | sort -u | head -5
      return 1
    fi
    sleep 5
  done
  [[ "${ready}" == true ]] || return 1
  echo "server ready (${arm})"

  # Residency and slot, so the planner's cost is on the record next to the
  # accuracy it bought.
  grep -hoE "Tiered MoE residency: .{0,80}|cold staging: .{0,80}|cold prefetch: [0-9]+ layers.{0,60}" \
    "${result_dir}/p6-${arm}-server."{out,err} 2>/dev/null | sort -u | head -4 || true

  # Two questions first: a transport fault scores zero in a few seconds and
  # is indistinguishable from a real result once written to the file.
  .venv/bin/python "${result_dir}/gsm8k_paired.py" \
    --num-questions 2 --pad-chars 36000 --num-shots 5 --max-tokens 256 --port 8027 \
    --out "${result_dir}/p6-${arm}-smoke.json"

  # Per-question outcomes, so the arms can be compared pairwise on the
  # identical question set rather than as two independent accuracies.
  .venv/bin/python "${result_dir}/gsm8k_paired.py" \
    --num-questions 1000 --pad-chars 36000 --num-shots 5 --max-tokens 256 --port 8027 \
    --out "${result_dir}/p6-${arm}-gsm8k.json"

  echo "--- chunks per request (expect ~2x questions on the staged arm) ---"
  grep -hc "staged [0-9.]* GiB across" \
    "${result_dir}/p6-${arm}-server."{out,err} 2>/dev/null | paste -sd+ || true

  echo "--- staging engagement (${arm}) ---"
  grep -hoE "cold prefetch: chunk .{0,150}" \
    "${result_dir}/p6-${arm}-server."{out,err} 2>/dev/null | tail -2 || true
  # The baseline arm legitimately has no such lines, and under
  # `set -o pipefail` an empty grep would otherwise kill the job.

  # Prefill throughput on the long prompt, same measurement as phase 3.
  ARM="${arm}" RESULT_DIR="${result_dir}" .venv/bin/python - <<'PY'
import json, os, time, urllib.request
from pathlib import Path
arm, out = os.environ["ARM"], Path(os.environ["RESULT_DIR"])
prompt = json.loads(Path(
    "agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl"
).read_text().splitlines()[0])["prompt"]
url = "http://127.0.0.1:8027/v1/completions"
def ask(n):
    body = json.dumps({"model": "glm53-cmp-tiered", "prompt": prompt,
                       "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()
    return time.perf_counter() - t
ask(4)
times = [ask(32) for _ in range(3)]
(out / f"p6-{arm}-prefill.json").write_text(json.dumps({"arm": arm, "seconds": times}))
print(f"{arm} prefill 96K x3: " + ", ".join(f"{t:.2f}s" for t in times))
PY

  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 20
}

run_arm baseline 0
run_arm staged 1024

echo "=== summary ==="
.venv/bin/python "${result_dir}/analyze_phase5.py" --prefix p6 || true
echo "=== done ==="
