#!/usr/bin/env bash
# Phase 5: A/B the cold prefetch on accuracy and throughput.
#
# Token comparison cannot gate this change -- phase 3b measured a run-to-run
# noise floor of 0.6-2.3 nats on the top-20 logprobs, in the baseline too. An
# eval is the right instrument: accuracy over many samples is stable where any
# single completion is not.
#
# MIN_TOKENS is 256 here, not the 1024 of phases 3/3b: GSM8K's 5-shot prompts
# measure ~539 tokens, so a 512 threshold would have been marginal and the
# staged path might never have engaged. Decode (4 tokens under MTP3) stays
# below 256 either way. The per-chunk log is checked to confirm engagement --
# an eval that silently exercised nothing is the failure mode here.
#
# Staging 48.5 GiB behind a ~539-token prefill is pure overhead in this regime;
# that is deliberate. This run gates correctness, not speed, and short prompts
# are the harsher test of it. The speed case is the 2026-09-04 roofline.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --job-name=cpfphase5
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
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-cpf5"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-cpf5"
export TRITON_CACHE_DIR="${cache_root}/triton-cpf5"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-cpf5"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-cpf5"
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
    >"${result_dir}/p5-${arm}-server.out" \
    2>"${result_dir}/p5-${arm}-server.err" &
  local pid=$!

  local ready=false
  for _ in $(seq 1 400); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ready=true; break; }
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SERVER EXITED EARLY (${arm})"
      grep -ohE "CUDA out of memory[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
        "${result_dir}/p5-${arm}-server."{out,err} 2>/dev/null \
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
    "${result_dir}/p5-${arm}-server."{out,err} 2>/dev/null | sort -u | head -4

  # Two questions first: a transport fault scores zero in a few seconds and
  # is indistinguishable from a real result once written to the file.
  .venv/bin/python "${result_dir}/gsm8k_paired.py" \
    --num-questions 2 --num-shots 5 --max-tokens 256 --port 8027 \
    --out "${result_dir}/p5-${arm}-smoke.json"

  # Per-question outcomes, so the arms can be compared pairwise on the
  # identical question set rather than as two independent accuracies.
  .venv/bin/python "${result_dir}/gsm8k_paired.py" \
    --num-questions 300 --num-shots 5 --max-tokens 256 --port 8027 \
    --out "${result_dir}/p5-${arm}-gsm8k.json"

  echo "--- staging engagement (${arm}) ---"
  grep -hoE "cold prefetch: chunk .{0,150}" \
    "${result_dir}/p5-${arm}-server."{out,err} 2>/dev/null | tail -2

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
(out / f"p5-{arm}-prefill.json").write_text(json.dumps({"arm": arm, "seconds": times}))
print(f"{arm} prefill 96K x3: " + ", ".join(f"{t:.2f}s" for t in times))
PY

  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 20
}

run_arm baseline 0
run_arm staged 256

echo "=== summary ==="
.venv/bin/python "${result_dir}/analyze_phase5.py" || true
echo "=== done ==="
