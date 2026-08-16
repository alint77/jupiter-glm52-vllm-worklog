#!/usr/bin/env bash
# Does the NCCL symmetric-memory pool grow asymmetrically across DCP4 ranks?
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=symm-mem-dcp2
#SBATCH --output=agent_space/experiments/2026-08-05-symm-mem-dcp/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-symm-mem-dcp/slurm-%x-%j.err
set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-05-symm-mem-dcp"
prompts="${repo_dir}/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/prompts.jsonl"
cd "${repo_dir}"
source agent_space/jupiter-env.sh
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
[[ -d "${model}" ]] || model="${repo_dir}/../models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
echo "node: $(hostname)"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
# The whole point: turn symmetric memory on under DCP4. The all-reduce path
# stays blocked by 990b1d378's guard; the all-gather/reduce-scatter path does
# not, so windows still get registered and any divergence is visible.
export VLLM_USE_NCCL_SYMM_MEM=1
# Lift 990b1d378's guard: this is the path it actually disables.
export VLLM_NCCL_SYMM_MEM_ALLOW_DCP=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-prod-profile"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-prod-profile"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-31-replica-scheduling-v2/hybrid-p0.5-replicas-985.json"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

arm="${ARM:-ag_rs}"
extra=()
[[ "${arm}" == "a2a" ]] && extra=(--dcp-comm-backend a2a)
echo "arm: ${arm} (symm-mem AR guard lifted)"
agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  "${extra[@]}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --tiered-moe-replica-assignment exact \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  >"${result_dir}/${arm}-server.out" 2>"${result_dir}/${arm}-server.err" &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT

ready=false
for _ in $(seq 1 400); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then echo "SERVER DIED"; break; fi
  sleep 5
done
if [[ "${ready}" != true ]]; then
  echo "NOT READY (a hang here is itself the result)"
  tail -30 "${result_dir}/${arm}-server.err"
else
  curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    >"${result_dir}/${arm}-semantic.json" || echo "SMOKE FAILED/HUNG"
  # Mixed shapes exercise the allocator the way DCP does in production.
  .venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered --tokenizer "${model}" \
    --dataset-name custom --dataset-path "${prompts}" --custom-output-len 64 \
    --disable-shuffle --num-prompts 8 --max-concurrency 4 --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm >/dev/null 2>&1 \
    || echo "BENCH FAILED/HUNG"
fi

echo ""
echo "=== per-rank symmetric-memory window registrations ==="
for r in 0 1 2 3; do
  n=$(grep -ac "Worker_TP${r}.*registering window" "${result_dir}/symm-server.out" \
        "${result_dir}/${arm}-server.err" 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  echo "rank ${r}: ${n} registrations"
done
grep -ah "registering window" "${result_dir}/symm-server.out" "${result_dir}/${arm}-server.err" \
  2>/dev/null | sed 's/.*\(Worker_TP[0-9]\).*registering window #\([0-9]*\) at \(0x[0-9a-f]*\), \([0-9]*\).*/\1 win\2 \3 \4B/' \
  | sort | tee "${result_dir}/${arm}-registrations.txt" | head -60
echo "=== done ==="
