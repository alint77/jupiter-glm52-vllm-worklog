#!/usr/bin/env bash
# Nsight Systems prefill-size arm at the c4 launcher config (MTP3, 4 seqs,
# pool 2) with VLLM_SCHED_INTERLEAVE_PREFILL=1 and the given token budget.
#   ./onnode.sh "<D>/arm.sh <tag> <budget> <sizes> [env...]"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-prefill-sizes
tag=$1 budget=$2 sizes=$3; shift 3
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}/trace"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS=
export SPEC=mtp SPEC_K=3 COMPILE_SIZES= MAX_NUM_SEQS=4 VLLM_TIERED_MOE_KV_POOL_SEQS=2 RESERVE_GB=3.6
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,64,128,256,384,512,640,768,896,1024
export VLLM_SCHED_INTERLEAVE_PREFILL=1 MAX_NUM_BATCHED_TOKENS=${budget}
for kv in "$@"; do export "${kv}"; done
export TRACE_ROOT="${OUT}/trace" NSYS_OUT="${OUT}/nsys" NSYS_BIN=${PWD}/agent_space/experiments/2026-10-09-tp-sliced-experts/nsys_node.sh
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; sleep 30; pkill -9 -f "VLLM::[W]orker" 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T) budget=${budget}"
grep -ah "observed HBM" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
.venv/bin/python ${D}/prof_sizes.py --sizes "${sizes}" --out "${OUT}/timings.jsonl"
sleep 20
ls "${OUT}"/nsys* 2>/dev/null | wc -l
echo "=== done $(date +%T)"
