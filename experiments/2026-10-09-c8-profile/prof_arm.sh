#!/usr/bin/env bash
# c=8, 1.6M KV pool (VLLM_TIERED_MOE_KV_POOL_SEQS=4), SPEC_K=3: torch-profiler
# windows (prof_load.py) and allocator snapshots (MEM_SNAPSHOT_DIR probe in
# gpu_worker.py), per-second nvidia-smi memory.
#   ./onnode.sh "<D>/prof_arm.sh <tag> <spec>"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-c8-profile
tag=$1 spec=$2
OUT=/e/fscratch/profound/${USER}/c8-profile/${tag}; mkdir -p "${OUT}/mem"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export SPEC=${spec} SPEC_K=3 MAX_NUM_SEQS=8 VLLM_TIERED_MOE_KV_POOL_SEQS=4 COMPILE_SIZES=
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,20,24,28,32,64,128,256,384,512,640,768,896,1024
[[ ${spec} == mtp ]] && export RESERVE_GB=3.6
export TRACE_ROOT="${OUT}/trace" MEM_SNAPSHOT_DIR="${OUT}/mem"; mkdir -p "${TRACE_ROOT}"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
grep -ah "observed HBM\|GPU KV cache size\|Maximum concurrency\|Tiered KV pool\|Tiered KV main\|Tiered KV draft\|Tiered KV indexer" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
grep -a "residency:" "${OUT}/server.out" | grep Worker_TP0 | tail -1 | sed 's/^.*\] //' | cut -c1-90
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem.csv" &
smi=$!
.venv/bin/python ${D}/prof_load.py --trace-root "${TRACE_ROOT}"
kill ${smi}
awk -F, '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "gpu %s peak %d MiB\n", g, m[g]}' "${OUT}/mem.csv"
echo "=== done $(date +%T)"
