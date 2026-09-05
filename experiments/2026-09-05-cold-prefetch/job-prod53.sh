#!/usr/bin/env bash
# Prod (GLM-5.3 W4A16, DFlash2 c=1) cold-prefetch before/after in ONE
# allocation, so node and code are held fixed and only MIN_TOKENS differs.
# Arm 1 boots prod's exact config with the flag and no profiler: the planner
# probe shows the budget fits, but only the runtime reserve check -- which the
# c1-df2 launcher notes is tight, with the drafter's unbudgeted ~2.18 GiB
# alongside the slot -- can prove it starts.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=prod53cpf
#SBATCH --output=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
input_dir="${repo_dir}/agent_space/experiments/2026-09-04-mtp3-profile"
result_dir="${repo_dir}/agent_space/experiments/2026-09-05-cold-prefetch"
prompts="${input_dir}/prompts.jsonl"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16"
drafter="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-DFlash2"
for d in "${model}" "${drafter}"; do
  [[ -d "${d}" ]] || { echo "missing: ${d}" >&2; exit 1; }
done
echo "node: $(hostname)"

# Prod's serving environment, verbatim from 2026-09-04-glm53-c1-df2.
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=10
export VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=0
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
cache_root="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin"
export VLLM_CACHE_ROOT="${cache_root}/vllm-cache-prod53"
export TRTLLM_DG_CACHE_DIR="${cache_root}/trtllm-dg-prod53"
export TRITON_CACHE_DIR="${cache_root}/triton-prod53"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor-prod53"
export FLASHINFER_CACHE_DIR="${cache_root}/flashinfer-prod53"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"

spec="{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}"

server_pid=""
cleanup() { [[ -n "${server_pid}" ]] && kill "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# $1 tag, $2 MIN_TOKENS, $3 "trace"|"notrace", $4 prefix-caching flag
start_server() {
  local tag="$1" min="$2" mode="$3" pc="$4"
  export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="${min}"
  local extra=()
  if [[ "${mode}" == trace ]]; then
    trace_root="/e/project1/profound/alint77/traces/prod53-${tag}-${SLURM_JOB_ID}"
    mkdir -p "${trace_root}"
    extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_root}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}")
  fi
  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config "${spec}" \
    --decode-context-parallel-size 1 \
    --max-num-seqs 1 \
    --served-model-name glm53-w4a16-tiered \
    --gpu-memory-utilization 0.90 \
    --max-model-len 400000 \
    "${pc}" \
    --generation-config vllm \
    "${extra[@]}" \
    >"${result_dir}/prod53-${tag}-server.out" \
    2>"${result_dir}/prod53-${tag}-server.err" &
  server_pid=$!
  local ready=false
  for _ in $(seq 1 360); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "=== ${tag}: SERVER DIED ==="
      grep -ohE "CUDA out of memory[^\"]{0,140}|reserve[^\"]{0,160}|[A-Za-z]*(Error|Exception): .{0,160}" \
        "${result_dir}/prod53-${tag}-server."{out,err} 2>/dev/null \
        | grep -viE "Engine core init|WorkerProc init" | sort -u | head -5
      exit 1
    fi
    sleep 5
  done
  [[ "${ready}" == true ]]
  echo "=== ${tag}: server ready ==="
}

stop_server() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  server_pid=""
  sleep 20
}

# Lines that prove the feature is live: the planner's budget, the runtime
# reserve check, and the per-chunk tier split.
report() {
  local tag="$1"
  echo "--- ${tag}: staging / reserve / tier split ---"
  grep -ohE "cold staging: .{0,90}|reserve.{0,120}|from slot.{0,40}|Tiered MoE residency: .{0,80}" \
    "${result_dir}/prod53-${tag}-server."{out,err} 2>/dev/null \
    | sort -u | head -12 || true
}

# ---------------- arm 1: prod config boots with the flag ----------------
start_server boot 1024 notrace --enable-prefix-caching
curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  >"${result_dir}/prod53-boot-smoke.json"
.venv/bin/python - "${prompts}" <<'PY' >"${result_dir}/prod53-boot-long.json"
import json, sys, urllib.request
p = json.loads(open(sys.argv[1]).readline())["prompt"]
req = urllib.request.Request("http://127.0.0.1:8027/v1/completions",
    data=json.dumps({"model": "glm53-w4a16-tiered", "prompt": p,
                     "max_tokens": 16, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=900))))
PY
report boot
stop_server

# ---------------- arms 2 and 3: traced pair, prefix caching off ----------
for arm in baseline staged; do
  min=0; [[ "${arm}" == staged ]] && min=1024
  start_server "${arm}" "${min}" trace --no-enable-prefix-caching
  curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm53-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    >"${result_dir}/prod53-${arm}-smoke.json"
  .venv/bin/python "${input_dir}/capture.py" --prompts "${prompts}" --trace-root "${trace_root}"
  cp "${trace_root}/capture-report.json" "${result_dir}/prod53-${arm}-capture-${SLURM_JOB_ID}.json"
  printf '%s\n' "${trace_root}" >"${result_dir}/prod53-${arm}-tracepath-${SLURM_JOB_ID}.txt"
  report "${arm}"
  stop_server
done

echo "=== done ==="
