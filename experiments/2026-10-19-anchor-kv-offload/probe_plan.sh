#!/usr/bin/env bash
# Planner-only probe at the c=4 launcher config (as nsys_arm_n.sh conc=4):
# start serve.sh, wait for TP0's residency line, kill. Prints hot/cold and
# the HBM available to experts.
#   probe_plan.sh <tag> [env...]   (on a node: onnode.sh "<this> <tag> ...")
set -uo pipefail
cd /e/project1/profound/alint77/vllm
tag=$1; shift
for kv in "$@"; do export "${kv}"; done
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS= SPEC=mtp SPEC_K=3 COMPILE_SIZES= MAX_NUM_SEQS=4
export VLLM_TIERED_MOE_KV_POOL_SEQS=${POOL_SEQS:-2} RESERVE_GB=${RESERVE_GB:-3.6}
sizes=$(seq 1 8; seq 12 4 16; echo 64 128 256 384 512 640 768 896 1024)
export CAPTURE_SIZES="$(tr ' ' '\n' <<<"${sizes}" | sort -nu | paste -sd,)"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
for _ in $(seq 1 120); do
  grep -aq "Worker_TP0.*residency:" "${OUT}/server.out" 2>/dev/null && break
  kill -0 "${pid}" 2>/dev/null || break
  sleep 5
done
# SIGTERM is not handled while the weights load
pkill -9 -f "bin/vllm [s]erve" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; kill -9 ${pid} 2>/dev/null; wait ${pid} 2>/dev/null
line=$(grep -a "Worker_TP0.*residency:" "${OUT}/server.out" | tail -1 | sed 's/^.*residency: //' | cut -c1-75)
echo "${tag}: ${line:-no residency line: $(grep -ohaE '[A-Za-z]*(Error|Exception): .{0,160}' "${OUT}"/server.* | sort -u | head -2)}"
