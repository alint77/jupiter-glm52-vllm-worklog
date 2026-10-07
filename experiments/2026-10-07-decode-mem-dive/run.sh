#!/usr/bin/env bash
# Prod serve.sh (agentic client flags) plus one of:
#   timing: agentic task set with torch-profiler windows, then long-context
#           decode windows (longctx.py)
#   nsys:   agentic task set, one Nsight Systems window, graphs traced whole
#   nsyslong: one Nsight Systems window in a 130K-token decode
#   mem:    MEM_SNAPSHOT_DIR probe (allocator snapshots with Python stacks at
#           startup / after profiling / after a 130K-token request)
#   ./onnode.sh <D>/run.sh <mode> <tag>
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-07-decode-mem-dive
B=agent_space/experiments/2026-09-28-agentic-decode-bench
mode=$1; tag=$2
OUT=/e/fscratch/profound/${USER}/decode-mem-dive/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TRACE_ROOT="${OUT}/trace"; mkdir -p "${TRACE_ROOT}"
case ${mode} in
  nsys|nsyslong) export NSYS_OUT="${OUT}/nsys" ;;
  mem) export MEM_SNAPSHOT_DIR="${OUT}/mem"; mkdir -p "${MEM_SNAPSHOT_DIR}" ;;
esac
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem.csv" &
smi=$!
name=glm53-w4a16-tiered
tasks=(agent_space/experiments/2026-09-26-mimo-routing-profile/tasks-{0,1,2,3}.json)
case ${mode} in
  timing)
    .venv/bin/python ${B}/agentic_bench.py --model ${name} --out "${OUT}/rows.jsonl" --tasks "${tasks[@]}" \
      --limit-requests 16 --profile 3 --trace-root "${TRACE_ROOT}"
    .venv/bin/python ${D}/longctx.py --tokens 50000,130000 --profile --trace-root "${TRACE_ROOT}" ;;
  nsys)
    .venv/bin/python ${B}/agentic_bench.py --model ${name} --out "${OUT}/rows.jsonl" --tasks "${tasks[@]}" \
      --limit-requests 12 --profile 1 --profile-window 3 --trace-root "${TRACE_ROOT}" ;;
  nsyslong)  # one Nsight window (the range ends at the first stop) at 130K
    .venv/bin/python ${D}/longctx.py --tokens 130000 --profile --trace-root "${TRACE_ROOT}" ;;
  mem)
    .venv/bin/python ${D}/longctx.py --tokens 130000 --max-tokens 300
    curl -fsS -X POST http://127.0.0.1:8027/start_profile; sleep 2
    curl -fsS -X POST http://127.0.0.1:8027/stop_profile ;;
esac
kill ${smi}
echo "=== done $(date +%T)"
