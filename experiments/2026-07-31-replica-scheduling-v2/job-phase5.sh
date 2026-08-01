#!/usr/bin/env bash
# Phase 5: paired same-node trace, to confirm the mechanism behind Phase 3.
#
# Phase 3 established that exact assignment makes decode 5-6% faster. This
# checks the reason is the one claimed - less rank skew, so less time waiting
# in the tensor-parallel all-reduce - rather than something incidental.
#
# Both arms trace on one node in one job. v1's trace pair was cross-node, and
# the nodes differ by ~10% on the fixed part of a Marlin call, which is what
# produced its phantom "Marlin counter-cost".
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --time-min=01:00:00
#SBATCH --job-name=replica-v2-phase5
#SBATCH --output=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.err

set -euo pipefail

seqs=4
repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replica-scheduling-v2"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
profile="${result_dir}/../2026-07-31-replicated-expert-scheduling/runtime-placements/replicas-985.json"
prompts="${repo}/agent_space/experiments/2026-07-29-marlin-smem-monopoly/agentic-prompts.jsonl"

cd "${repo}"
source agent_space/jupiter-env.sh

probe="/e/project1/profound/alint77/.replica-caches/.quota-probe-${SLURM_JOB_ID}"
mkdir -p "$(dirname "${probe}")"
if ! touch "${probe}" 2>/dev/null; then
  echo "project1 quota is exhausted; clear .replica-caches before running" >&2
  exit 1
fi
rm -f "${probe}"

echo "node: $(hostname)  job: ${SLURM_JOB_ID}"

for assignment in off exact; do
  tag="phase5-${SLURM_JOB_ID}-${assignment}"
  trace_dir="/e/scratch/profound/naeimitabiei1/replica-v2-${tag}"
  mkdir -p "${trace_dir}"
  printf '%s\n' "${trace_dir}" >"${result_dir}/${tag}-trace-path.txt"
  echo "=== arm ${assignment} -> ${trace_dir} ==="

  unset VLLM_USE_V2_MODEL_RUNNER
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.replica-caches/v2-s${seqs}-mtp3"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.replica-caches/v2t-s${seqs}-mtp3"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
  export VLLM_TIERED_MOE_PROFILE_CAP=1
  export TIERED_MOE_HBM_RESERVE_GB=7
  export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[4,8,12,16],\"compile_sizes\":[4,8,12,16],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  # 32 decode graphs per rank, against v1's 6: its trace was underpowered on
  # step time and could only speak to mechanism.
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":4,\"max_iterations\":40}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --tiered-moe-replica-assignment "${assignment}" \
    --tiered-moe-host-reserve-gb 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs "${seqs}" \
    --no-enable-prefix-caching \
    --decode-context-parallel-size 4 \
    --profiler-config "${profiler_config}" \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
  trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

  ready=false
  for _ in $(seq 1 900); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      ready=true; break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -60 "${result_dir}/${tag}-server.err" >&2
      exit 1
    fi
    sleep 5
  done
  [[ "${ready}" == true ]]

  bench_args=(
    --backend openai --base-url http://127.0.0.1:8027
    --endpoint /v1/completions --model glm52-w4a16-tiered
    --served-model-name glm52-w4a16-tiered --tokenizer "${model}"
    --dataset-name custom --dataset-path "${prompts}"
    --custom-output-len 256 --skip-chat-template --disable-shuffle
    --num-prompts $(( seqs * 4 )) --max-concurrency "${seqs}"
    --request-rate inf --temperature 0 --ignore-eos --disable-tqdm
  )
  .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null

  curl -fsS -X POST http://127.0.0.1:8027/start_profile >/dev/null
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result --result-dir "${result_dir}" --result-filename "${tag}.json"
  curl -fsS -X POST http://127.0.0.1:8027/stop_profile >/dev/null
  sleep 30

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 20
done

echo "=== census and skew ==="
.venv/bin/python "${result_dir}/analyze_marlin_overlap.py" \
  --arm off "/e/scratch/profound/naeimitabiei1/replica-v2-phase5-${SLURM_JOB_ID}-off" \
  --arm exact "/e/scratch/profound/naeimitabiei1/replica-v2-phase5-${SLURM_JOB_ID}-exact" \
  --output "${result_dir}/phase5-overlap-${SLURM_JOB_ID}.json"
