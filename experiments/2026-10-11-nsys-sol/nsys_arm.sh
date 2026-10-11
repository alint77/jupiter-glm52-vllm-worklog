#!/usr/bin/env bash
# Nsight Systems arm (graph kernels per node, one report per profiler window)
# at a Claude Code launcher's config, from the caller's PYTHONPATH tree:
#   c = 1   claude-glm53-sliced-df2-dcp4.sh (serve.sh defaults: DFlash2 k=7)
#   c > 1   claude-glm53-sliced-mtp3-c<c>.sh (MTP3, max_num_seqs c, captures
#           to 4c, pool and reserve as there)
# Windows: c requests at ~5K and at ~50K context (prof_load_n.py).
#   ./onnode.sh "<D>/nsys_arm.sh <tag> <c> [env...]"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-nsys-sol
tag=$1 conc=$2; shift 2
for kv in "$@"; do export "${kv}"; done
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}/trace"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS=
if (( conc > 1 )); then
  declare -A pool=([2]=2 [4]=2 [8]=4 [16]=4) reserve=([2]=3.6 [4]=3.6 [8]=3.6 [16]=4.0)
  export SPEC=mtp SPEC_K=3 COMPILE_SIZES= MAX_NUM_SEQS=${conc}
  export VLLM_TIERED_MOE_KV_POOL_SEQS=${pool[${conc}]} RESERVE_GB=${reserve[${conc}]}
  sizes=$(seq 1 8; seq 12 4 $((4 * conc)); echo 64 128 256 384 512 640 768 896 1024)
  export CAPTURE_SIZES="$(tr ' ' '\n' <<<"${sizes}" | sort -nu | paste -sd,)"
fi
export TRACE_ROOT="${OUT}/trace" NSYS_OUT="${OUT}/nsys" NSYS_BIN=${PWD}/agent_space/experiments/2026-10-09-tp-sliced-experts/nsys_node.sh
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
grep -ah "observed HBM\|GPU KV cache size\|Maximum concurrency" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
grep -a "residency:" "${OUT}/server.out" | grep Worker_TP0 | tail -1 | sed 's/^.*\] //' | cut -c1-90
.venv/bin/python ${D}/prof_load_n.py --n "${conc}" --trace-root "${TRACE_ROOT}"
sleep 20
ls -la "${OUT}"/nsys* 2>/dev/null
echo "=== done $(date +%T)"
