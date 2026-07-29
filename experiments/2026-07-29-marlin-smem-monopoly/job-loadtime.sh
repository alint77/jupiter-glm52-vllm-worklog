#!/usr/bin/env bash
# Checkpoint load time: exa_project1 vs exa_fscratch, on the real c4 serving path.
#
# This mirrors claude-local-c4.sh (the way the model is actually used): V2
# runner, DCP4, max_num_seqs 4, MTP3, tiered MoE. Only TIERED_MOE_MODEL_PATH
# differs between the two arms.
#
# Load is timed from the server log rather than wall clock, so compile and graph
# capture are excluded and only the weight-loading phase is compared. Both arms
# run in one allocation, project1 first, so the fscratch arm cannot benefit from
# anything the first arm warmed - they are different filesystems and different
# inodes, and 405 GB does not fit in the node's 857 GB alongside the model
# itself being resident.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=marlin-loadtime
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
name=GLM-5.2-AutoRound-W4G64-MTP-e1ba887

cd "${repo_dir}"
source agent_space/jupiter-env.sh
echo "node $(hostname)"

# arms to run, e.g. `sbatch job-loadtime.sh fscratch`. project1's rate is known
# from production runs (10.5-11.8 s/shard) and from this job's own first arm
# (19.1 s/shard under contention), so it does not need re-measuring every time.
for fs in ${1:-project1 fscratch}; do
  case "${fs}" in
    project1) model="/e/project1/profound/alint77/models/${name}" ;;
    fscratch) model="/e/fscratch/profound/naeimitabiei1/models/${name}" ;;
  esac
  if [[ ! -d "${model}" ]]; then
    echo "${fs}: ${model} missing, skipping"
    continue
  fi
  tag="loadtime-${fs}"

  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-loadtime"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-loadtime"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
  export VLLM_TIERED_MOE_PROFILE_CAP=0
  export VLLM_TIERED_MOE_TIGHT_SMEM=1
  export TIERED_MOE_HBM_RESERVE_GB=7
  export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  echo "=== ${fs}: ${model} ==="
  start=$(date +%s)
  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --decode-context-parallel-size 4 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
  cleanup() { kill "${server_pid}" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=false
  for _ in $(seq 1 900); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "${fs}: server died"
      tail -40 "${result_dir}/${tag}-server.err"
      break
    fi
    sleep 5
  done
  ready_at=$(date +%s)

  if [[ "${ready}" == true ]]; then
    echo "${fs}: healthy after $((ready_at - start))s total"
    # weight-loading phase only, as reported by the loader itself
    grep -oE "Loading weights took [0-9.]+ seconds" \
      "${result_dir}/${tag}-server.err" | tail -1 || true
    grep -oE "Model loading took [0-9.]+ (GiB|GB) and [0-9.]+ seconds" \
      "${result_dir}/${tag}-server.err" | tail -1 || true
    grep -oE "Loading safetensors checkpoint shards: 100%[^\r]*" \
      "${result_dir}/${tag}-server.err" | tail -1 || true
  fi

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 45
done
