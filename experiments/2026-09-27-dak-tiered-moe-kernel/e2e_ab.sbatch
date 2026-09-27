#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=mimo26-decode-ab
#SBATCH --output=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel/e2e-ab-%j.out
#SBATCH --error=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel/e2e-ab-%j.err

# Same-node end-to-end A/B of the tiered MoE decode kernel for MiMo-V2.6
# (serve.sbatch config: routing profile + 1500 replicas, DFlash k=7):
# "marlin" is production, "tiered" sets VLLM_TIERED_MOE_DECODE_KERNEL=1, and
# "prev" runs the tiered kernel from the worktree at PREV_WT (an earlier commit,
# build artifacts copied in) by shadowing the package with PYTHONPATH. Each arm
# restarts the server.
#
#   sbatch --export=ALL,ORDER="marlin tiered marlin tiered",GSM8K=400 e2e_ab.sbatch
#
# Decode is the bench.sbatch suite (16 coding prompts, 1024 output tokens,
# greedy, ignore-eos, c=1); GSM8K=N adds the quality check, SAMPLED=1 a second
# decode pass at temperature 1, top-p 0.95.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel
model=/e/fscratch/profound/${USER}/models/MiMo-V2.6-Pro-RL
out="${E}/e2e-ab-${SLURM_JOB_ID}"
mkdir -p "${out}"
export MIMO_PORT=8031
sed -e 's/^exec \.venv/.venv/' -e '/^#SBATCH/d' \
  agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch > "${out}/serve-inline.sh"

bench() {
  .venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8031 \
    --endpoint /v1/completions --model mimo26-pro --served-model-name mimo26-pro \
    --tokenizer "${model}" --trust-remote-code \
    --max-concurrency 1 --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 "$@"
}

stop_server() {
  kill -TERM -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    (( used < 2000 )) && break
    sleep 5
  done
}

run=0
for arm in ${ORDER:-marlin tiered marlin tiered}; do
  run=$((run + 1))
  tag="${run}-${arm}"
  case "${arm}" in
    tiered) export VLLM_TIERED_MOE_DECODE_KERNEL=1; unset PYTHONPATH ;;
    marlin) export VLLM_TIERED_MOE_DECODE_KERNEL=0; unset PYTHONPATH ;;
    prev) export VLLM_TIERED_MOE_DECODE_KERNEL=1 PYTHONPATH="${PREV_WT:?}" ;;
    *) echo "unknown arm ${arm}"; exit 1 ;;
  esac
  setsid bash "${out}/serve-inline.sh" >"${out}/server-${tag}.out" 2>"${out}/server-${tag}.err" &
  pid=$!
  ready=false
  for _ in $(seq 1 180); do
    curl -fsS http://127.0.0.1:8031/health >/dev/null 2>&1 && { ready=true; break; }
    kill -0 "${pid}" 2>/dev/null || break
    sleep 5
  done
  if [[ "${ready}" != true ]]; then
    echo "${tag}: SERVER FAILED"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${out}/server-${tag}".{out,err} | sort -u | head -5
    stop_server
    continue
  fi
  echo "=== ${tag} ready at $(date +%T)"
  bench --dataset-name random --random-input-len 512 --random-output-len 64 \
    --num-prompts 2 >/dev/null 2>&1 || true
  bench --dataset-name custom \
    --dataset-path agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl \
    --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
    --save-result --result-dir "${out}" --result-filename "decode-${tag}.json" 2>&1 \
    | grep -E "Mean TPOT|Median TPOT|Output token throughput|acceptance"
  if [[ -n "${SAMPLED:-}" ]]; then
    # the chat default (temperature 1, top-p 0.95): exercises the sampler
    bench --dataset-name custom \
      --dataset-path agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl \
      --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
      --temperature 1.0 --top-p 0.95 \
      --save-result --result-dir "${out}" --result-filename "sampled-${tag}.json" 2>&1 \
      | grep -E "Median TPOT|Output token throughput"
  fi
  curl -fsS http://127.0.0.1:8031/metrics 2>/dev/null \
    | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:spec_decode_num_drafts_total" \
    > "${out}/spec-${tag}.txt" || true
  if [[ -n "${GSM8K:-}" ]]; then
    TMPDIR=/e/fscratch/profound/${USER}/caches/gsm8k \
      .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 \
      --port 8031 --num-questions "${GSM8K}" --temperature 0 \
      --save-results "${out}/gsm8k-${tag}.json" >"${out}/gsm8k-${tag}.log" 2>&1 || true
    grep -iE "^accuracy|^invalid" "${out}/gsm8k-${tag}.log" || tail -3 "${out}/gsm8k-${tag}.log"
  fi
  stop_server
done
echo "AB DONE"
