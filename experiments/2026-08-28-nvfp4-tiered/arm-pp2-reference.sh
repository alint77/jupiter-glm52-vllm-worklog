#!/usr/bin/env bash
# Reference arm: GLM-5.3 NVFP4 with the tiered MoE path OFF, across two nodes.
#
# The hypothesis under test is that the tiered MoE implementation degrades
# model quality. The sharpest way to test it is to hold everything else fixed
# -- same binary, same checkpoint, same quantization, same kernels -- and vary
# only whether the tiered path is used. Tiered off does not fit on one node
# (433 GiB against 4 x 95), so the reference needs two: PP2 splits the 78
# layers into two stages and TP4/EP4 shards within each, giving 54 GiB/GPU
# with ~41 GiB left for KV and activations.
#
# `validate_tiered_moe` returns immediately when tiered is disabled, so its
# PP1 requirement does not apply here.
#
# Comparison is teacher-forced: `prompt_logprobs` over a fixed prompt gives a
# distribution at every token position of an identical input, so nothing can
# diverge the way sampled generations do. Comparing generations would only
# re-measure float non-determinism.

set -euo pipefail

label="${1:?label}"
head_node="${2:?head node hostname}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node:  $(hostname)   head: ${head_node}"
echo "label: ${label}"

model="/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-nvfp4-pp2"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-nvfp4-pp2"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

# Ray is vLLM's multi-node executor. The head must be reachable from both
# nodes; Booster interconnect hostnames resolve inside the allocation.
export RAY_ADDRESS="${head_node}:6379"
export VLLM_HOST_IP="$(hostname -I | awk '{print $1}')"

if [[ "$(hostname -s)" == "${head_node}" ]]; then
  ray start --head --port=6379 --num-gpus=4 --disable-usage-stats \
    >"${result_dir}/${label}-rayhead.log" 2>&1
else
  sleep 20
  ray start --address="${head_node}:6379" --num-gpus=4 --disable-usage-stats \
    >"${result_dir}/${label}-rayworker.log" 2>&1
  # Workers only join the cluster; the server is launched from the head.
  sleep infinity
fi

# Wait for both nodes to register before serving.
for _ in $(seq 1 60); do
  n=$(ray status 2>/dev/null | grep -cE "^ *1 node_|GPU" || echo 0)
  gpus=$(python - <<'PY' 2>/dev/null || echo 0
import ray, re
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
)
  [[ "${gpus}" -ge 8 ]] && { echo "ray cluster ready: ${gpus} GPUs"; break; }
  sleep 5
done

.venv/bin/vllm serve "${model}" \
  --served-model-name glm53-nvfp4-ref \
  --host 0.0.0.0 --port 8033 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 64 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  >"${result_dir}/${label}-server.out" 2>"${result_dir}/${label}-server.err" &
pid=$!

ready=false
for _ in $(seq 1 500); do
  curl -fsS http://127.0.0.1:8033/health >/dev/null 2>&1 && { ready=true; break; }
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,150}" \
      "${result_dir}/${label}-server."{out,err} 2>/dev/null \
      | grep -viE "Engine core init|WorkerProc init" | sort -u | head -4
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]] || { echo "SERVER NOT READY"; kill "${pid}" 2>/dev/null; exit 1; }
echo "server ready (PP2 x TP4, tiered OFF)"

.venv/bin/python "${result_dir}/capture_logprobs.py" \
  --label "${label}" --port 8033 --model glm53-nvfp4-ref \
  --out "${result_dir}/${label}-logprobs.json"

kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
ray stop >/dev/null 2>&1 || true
