#!/usr/bin/env bash
# Phase 3: same-node A/B of exact replica assignment.
#
# Two regimes. With spec=off there is no speculation, so the decode batch is
# exactly the sequence count and draft acceptance cannot move between arms:
# mean TPOT is the step time directly. With spec=mtp3 the batch is 4x larger,
# which is the production shape, but acceptance returns as a confound - its
# paired delta across three v1 same-node experiments was -8.45%, -2.75% and
# +2.63%, a sign that does not reproduce - so there the headline is the
# acceptance-corrected step time.
#
# The plan asked for an acceptance-free run at batch 16. That is not reachable:
# the tiered path caps max_num_seqs at 4 because the replicated 400K MLA cache
# does not fit more per rank, so batch 16 requires MTP. The acceptance-free
# arm therefore runs at batch 4, and batch 16 is covered with MTP plus the
# device-side trace, which is acceptance-independent by construction.
#
# Both arms run in one job on one node, alternating, because every v1 A/B pair
# - including its traces - was cross-node, and the nodes differ by ~10% on the
# fixed part of a Marlin call.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --time-min=01:30:00
#SBATCH --job-name=replica-v2-phase3
#SBATCH --output=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.err

set -euo pipefail

# Sequence slots. The tiered path caps this at 4: the replicated 400K MLA
# cache does not fit more sequences per rank, and above 1 it requires DCP.
seqs="${1:-4}"
# "off" runs without speculation, so the decode batch is exactly `seqs` and
# acceptance cannot vary between arms. "mtp3" is the production shape, decode
# batch 4 x seqs, where acceptance-corrected step time is the headline.
spec="${2:-off}"
repeats="${3:-5}"

repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replica-scheduling-v2"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
profile="${result_dir}/../2026-07-31-replicated-expert-scheduling/runtime-placements/replicas-1697.json"
prompts="${repo}/agent_space/experiments/2026-07-29-marlin-smem-monopoly/agentic-prompts.jsonl"

cd "${repo}"
source agent_space/jupiter-env.sh
verify_tokens=1
spec_args=()
if [[ "${spec}" == "mtp3" ]]; then
  verify_tokens=4
  spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi
batch=$(( seqs * verify_tokens ))
tag_base="phase3-${SLURM_JOB_ID}-s${seqs}-${spec}"
echo "node: $(hostname)  job: ${SLURM_JOB_ID}  seqs: ${seqs}  spec: ${spec}"
echo "decode batch: ${batch}  repeats: ${repeats}"

# Capture every step size the batch can take: MTP verifies 4 tokens per
# sequence but a draining batch produces smaller graphs.
graph_sizes="${batch}"
if (( verify_tokens > 1 )); then
  graph_sizes=""
  for size in $(seq "${verify_tokens}" "${verify_tokens}" "${batch}"); do
    graph_sizes="${graph_sizes:+${graph_sizes},}${size}"
  done
fi

dcp_args=()
if (( seqs > 1 )); then
  # Required above one sequence, not a tuning choice.
  dcp_args=(--decode-context-parallel-size 4)
fi

start_server() {
  local assignment="$1" tag="$2"
  unset VLLM_USE_V2_MODEL_RUNNER
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.replica-caches/v2-${assignment}-s${seqs}-${spec}"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.replica-caches/v2t-${assignment}-s${seqs}-${spec}"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
  export VLLM_TIERED_MOE_PROFILE_CAP=1
  export TIERED_MOE_HBM_RESERVE_GB=7
  export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${graph_sizes}],\"compile_sizes\":[${graph_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    "${spec_args[@]}" \
    --tiered-moe-replica-assignment "${assignment}" \
    --tiered-moe-host-reserve-gb 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs "${seqs}" \
    --no-enable-prefix-caching \
    "${dcp_args[@]}" \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
}

wait_ready() {
  local tag="$1"
  for _ in $(seq 1 900); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "server for ${tag} died before readiness" >&2
      tail -80 "${result_dir}/${tag}-server.err" >&2
      return 1
    fi
    sleep 5
  done
  echo "server for ${tag} never became ready" >&2
  return 1
}

# Arms alternate within the job so slow drift cannot align with one arm.
for round in $(seq 1 "${repeats}"); do
  for assignment in off exact; do
    tag="${tag_base}-${assignment}-r${round}"
    echo "=== round ${round}, arm ${assignment} ==="
    start_server "${assignment}" "${tag}"
    trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
    wait_ready "${tag}"

    curl -fsS http://127.0.0.1:8027/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}' \
      >"${result_dir}/${tag}-smoke.json"

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
      --num-prompts $(( seqs * 4 ))
      --max-concurrency "${seqs}"
      --request-rate inf
      --temperature 0
      --ignore-eos
      --disable-tqdm
    )
    # Warm the graph and the Grace pages before the measured run.
    .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null

    curl -fsS http://127.0.0.1:8027/metrics \
      >"${result_dir}/${tag}-metrics-before.txt"
    .venv/bin/vllm bench serve "${bench_args[@]}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${tag}.json"
    curl -fsS http://127.0.0.1:8027/metrics \
      >"${result_dir}/${tag}-metrics-after.txt"

    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    trap - EXIT
    sleep 20
  done
done

echo "=== summary ==="
.venv/bin/python "${result_dir}/summarize_phase3.py" \
  --result-dir "${result_dir}" \
  --job "${SLURM_JOB_ID}" \
  --seqs "${seqs}" \
  --spec "${spec}" \
  --output "${result_dir}/phase3-summary-${SLURM_JOB_ID}.json"
