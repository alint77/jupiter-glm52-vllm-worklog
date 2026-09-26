#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-place-ab
#SBATCH --output=agent_space/experiments/2026-09-26-mimo-routing-profile/ab-%j.out
#SBATCH --error=agent_space/experiments/2026-09-26-mimo-routing-profile/ab-%j.err

# Same-node decode A/B of expert placement for MiMo-V2.6 (serve.sbatch config,
# DFlash k=7): "linear" is today's profile-less even split, "profile" adds
# --tiered-moe-placement-profile. Each arm restarts the server, so the arms
# differ only in which experts are HBM-resident.
#
#   sbatch --export=ALL,PROFILE=<json>,ORDER="linear profile linear profile" ab.sbatch
#
# Decode is the bench.sbatch suite: 16 PyTorch coding prompts, 1024 output
# tokens, greedy, ignore-eos, c=1. None of these prompts were in the capture.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-26-mimo-routing-profile
model=/e/fscratch/profound/${USER}/models/MiMo-V2.6-Pro-RL
profile="${PROFILE:?set PROFILE}"
out="${E}/ab-${SLURM_JOB_ID}"
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

# setsid puts each server in its own process group: killing the wrapper bash
# alone would leave `vllm serve` and its workers holding the GPUs.
stop_server() {
  kill -TERM -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  # The next arm sizes residency from observed free HBM, so wait for it.
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    (( used < 2000 )) && break
    sleep 5
  done
}

run=0
for arm in ${ORDER:-linear profile linear profile}; do
  run=$((run + 1))
  tag="${run}-${arm}"
  if [[ "${arm}" == profile ]]; then
    export MIMO_EXTRA_ARGS="--language-model-only --tiered-moe-placement-profile ${profile}"
  else
    export MIMO_EXTRA_ARGS="--language-model-only"
  fi
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
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${out}/server-${tag}".{out,err} \
      | sort -u | head -5
    stop_server
    continue
  fi
  echo "=== ${tag} ready at $(date +%T)"
  tr '\r' '\n' < "${out}/server-${tag}.out" | grep -ohE "Tiered MoE residency:.{0,160}" | sort -u
  bench --dataset-name random --random-input-len 512 --random-output-len 64 \
    --num-prompts 2 >/dev/null 2>&1 || true
  bench --dataset-name custom \
    --dataset-path agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl \
    --custom-output-len 1024 --disable-shuffle --num-prompts 16 \
    --save-result --result-dir "${out}" --result-filename "decode-${tag}.json" 2>&1 \
    | grep -E "Mean TPOT|Median TPOT|Output token throughput|acceptance"
  curl -fsS http://127.0.0.1:8031/metrics 2>/dev/null \
    | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:spec_decode_num_drafts_total" \
    > "${out}/spec-${tag}.txt" || true
  # GSM8K=N adds a quality check: placement moves which kernel reads an
  # expert, so it should be accuracy-neutral.
  if [[ -n "${GSM8K:-}" ]]; then
    .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 \
      --port 8031 --num-questions "${GSM8K}" --temperature 0 \
      --save-results "${out}/gsm8k-${tag}.json" 2>&1 | grep -iE "accuracy|invalid"
  fi
  stop_server
done
echo "AB DONE"
