#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-bench
#SBATCH --output=agent_space/experiments/2026-09-22-mimo-v26-pro/bench-%j.out
#SBATCH --error=agent_space/experiments/2026-09-22-mimo-v26-pro/bench-%j.err

# Serve MiMo-V2.6-Pro-RL on tiered MoE and measure it in the same allocation.
#
#   decode   the PyTorch coding suite the GLM numbers use (16 prompts, 1024
#            output tokens, ignore-eos), concurrency 1, no speculative decoding
#   prefill  random prompts at 32K and 128K input, 64 output tokens, so TTFT
#            is essentially prefill time
#
# The server command is serve.sbatch's, run in the background here instead of
# exec'd, so the two cannot drift.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-22-mimo-v26-pro
model=/e/fscratch/profound/${USER}/models/MiMo-V2.6-Pro-RL
result_dir="${E}/bench-${SLURM_JOB_ID}"
mkdir -p "${result_dir}"
export MIMO_PORT=8031

sed -e 's/^exec \.venv/.venv/' -e '/^#SBATCH/d' "${E}/serve.sbatch" > "${result_dir}/serve-inline.sh"
bash "${result_dir}/serve-inline.sh" >"${result_dir}/server.out" 2>"${result_dir}/server.err" &
pid=$!

ready=false
for _ in $(seq 1 540); do
  curl -fsS http://127.0.0.1:8031/health >/dev/null 2>&1 && { ready=true; break; }
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,180}" \
      "${result_dir}/server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -5
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]] || { echo "SERVER NOT READY"; kill "${pid}"; exit 1; }
echo "server ready at $(date +%T)"
tr '\r' '\n' < "${result_dir}/server.out" | grep -ohE "Tiered MoE residency:.{0,70}|observed HBM reserve:.{0,50}|GPU KV cache size:.{0,25}|conversion.{0,60}" | sort -u

bench() {
  .venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8031 \
    --endpoint /v1/completions --model mimo26-pro --served-model-name mimo26-pro \
    --tokenizer "${model}" --trust-remote-code \
    --max-concurrency 1 --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 "$@"
}
reset() { curl -fsS -X POST http://127.0.0.1:8031/reset_prefix_cache >/dev/null || true; }

echo "=== warmup (excluded) ==="
bench --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 2 >/dev/null 2>&1 || true
reset

echo "=== decode: PyTorch coding suite, c=1, 1024 out ==="
bench --dataset-name custom \
  --dataset-path agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl \
  --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
  --save-result --result-dir "${result_dir}" --result-filename decode.json 2>&1 | tail -30
reset

for len in 32768 131072; do
  echo "=== prefill: random ${len} in, 64 out ==="
  bench --dataset-name random --random-input-len "${len}" --random-output-len 64 \
    --num-prompts 2 --save-result --result-dir "${result_dir}" \
    --result-filename "prefill-${len}.json" 2>&1 | tail -30
  reset
done

kill "${pid}" 2>/dev/null || true
echo "BENCH DONE"
