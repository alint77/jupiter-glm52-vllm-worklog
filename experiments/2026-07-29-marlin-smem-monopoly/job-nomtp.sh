#!/usr/bin/env bash
# Isolate the kernel effect from MTP acceptance.
#
# With speculative decoding on, throughput conflates step time with draft
# acceptance, and the two arms did not accept equally (2.889 vs 2.959 tokens per
# target step). Without MTP each step emits exactly one token per sequence, so
# mean TPOT *is* the step time and there is no acceptance term to divide out.
# Batch size then comes from concurrency. The tiered contract caps max_num_seqs
# at 4, so this covers M = 1, 2, 4 - M=4 being the same decode batch that MTP3
# at concurrency 1 produces, but with independent sequences and no acceptance.
#
# Note this is a different routing regime, not just a quieter one: M=4 from four
# independent sequences activates more distinct experts than M=4 from four
# consecutive positions of one sequence. It is a diagnostic; the MTP3 run stays
# the production gate.
#
# Both arms run sequentially in one allocation, so node-to-node variation cannot
# explain the difference either.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=marlin-smem-nomtp
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node: $(hostname)"

for label in off on; do
  case "${label}" in
    on) tight=1 ;;
    off) tight=0 ;;
  esac
  tag="nomtp-${label}"

  # M>1 without speculation needs the c4 shape: the tiered contract rejects
  # max_num_seqs>1 without DCP because the replicated 400K MLA cache does not
  # fit more than one sequence. This is the qualified c4 config minus MTP.
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-marlin-nomtp-${label}"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-marlin-nomtp-${label}"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
  export VLLM_TIERED_MOE_PROFILE_CAP=0
  export VLLM_TIERED_MOE_TIGHT_SMEM="${tight}"
  export TIERED_MOE_HBM_RESERVE_GB=7
  # No speculation, so the decode batch is the concurrency. The tiered contract
  # pins max_num_seqs to 1..4 (vllm/config/vllm.py:2302), so M tops out at 4 -
  # which is exactly the c1/q4 decode batch this deployment runs. Capture every
  # concurrency benchmarked; tiered_overlap_max_tokens is 1 x 4 = 4, so the
  # overlap path and the launch policy cover all of them.
  export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4],"compile_sizes":[1,2,4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --decode-context-parallel-size 4 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
  cleanup() { kill "${server_pid}" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=false
  for _ in $(seq 1 720); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -60 "${result_dir}/${tag}-server.err"
      exit 1
    fi
    sleep 10
  done
  [[ "${ready}" == true ]]

  curl -fsS http://127.0.0.1:8027/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}' \
    >"${result_dir}/${tag}-smoke.json"
  python3 -c "
import json, sys
text = json.load(open('${result_dir}/${tag}-smoke.json'))['choices'][0]['text']
print('${tag} smoke:', repr(text))
sys.exit(0 if text == ' Paris. Distance from Paris to Lyon is' else 1)
"

  for conc in 1 2 4; do
    nprompts=$(( conc * 4 ))
    if (( nprompts < 24 )); then nprompts=24; fi
    bench_args=(
      --backend openai
      --base-url http://127.0.0.1:8027
      --endpoint /v1/completions
      --model glm52-w4a16-tiered
      --served-model-name glm52-w4a16-tiered
      --tokenizer "${model}"
      --dataset-name custom
      --dataset-path "${prompts}"
      --custom-output-len 256
      --skip-chat-template
      --disable-shuffle
      --num-prompts "${nprompts}"
      --max-concurrency "${conc}"
      --request-rate inf
      --temperature 0
      --ignore-eos
      --disable-tqdm
    )
    # one unrecorded warmup pass per concurrency, then two measured
    .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null
    for repeat in 1 2; do
      .venv/bin/vllm bench serve "${bench_args[@]}" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "${tag}-c${conc}-r${repeat}.json"
    done
  done

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 30
done
