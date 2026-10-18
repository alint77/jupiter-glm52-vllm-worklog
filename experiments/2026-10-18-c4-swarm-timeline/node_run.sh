#!/usr/bin/env bash
# One campaign arm on the node, prod c=4 config (claude-glm53-sliced-mtp3-c4.sh)
# from the frozen tree pf5 (vllm 144f2f4aee), with mem_sampler.py running from
# before the server starts (startup memory phases) to the end:
#   tlpyspy  c=4 timeline (timeline_run.py), no nsys; py-spy on the API server,
#            EngineCore and worker 0 for the whole replay
#   tlmem    the same without py-spy (overhead check, variance)
#   tlnsys   c=4 timeline with node-traced nsys windows (TIMELINE_WINDOWS)
#   dec<n>   n requests decoding at ~5K and ~50K (nsys_arm_n.sh, prof_load_n.py)
#   tlsnap   c=4 timeline on worktrees/pf5mem (allocator history + snapshots
#            at startup and at each TIMELINE_WINDOWS start/stop; perturbs timing)
#   sizes    pure prefill steps (prof_sizes.py), fresh and after a cached 60K
#     node_run.sh <tag> <kind> <compile cache> <triton cache> [env...]
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
tag=$1 kind=$2 cb=$3 tc=$4; shift 4
for kv in "$@"; do export "${kv}"; done
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}"
env=(PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/pf5 SERVE_CACHE_ROOT=${cb}
     TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false "$@")
.venv/bin/python ${D}/mem_sampler.py --out "${OUT}/mem" &
ms=$!
trap 'kill ${ms} 2>/dev/null' EXIT
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
case ${kind} in
  tlpyspy)
    bash ${D}/pyspy_run.sh "${OUT}" "${TIMELINE_DURATION:-900}" &
    ${A} ${tag} script "${env[@]}" SCRIPT=${D}/timeline_run.py TIMELINE_NOPROF=1 ;;
  tlmem) ${A} ${tag} script "${env[@]}" SCRIPT=${D}/timeline_run.py TIMELINE_NOPROF=1 ;;
  tlnsys) ${A} ${tag} script "${env[@]}" SCRIPT=${D}/timeline_run.py ;;
  dec*) agent_space/experiments/2026-10-17-dense-fusion-plan/nsys_arm_n.sh ${tag} 4 ${kind#dec} "${env[@]}" ;;
  tlsnap) mkdir -p "${OUT}/snap"
    ${D}/arm_snap.sh ${tag} script "${env[@]}" SCRIPT=${D}/timeline_run.py \
      PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/pf5mem MEM_SNAPSHOT_DIR=${OUT}/snap ;;
  sizes) ${A} ${tag} sizes "${env[@]}" ;;
esac
wait $(jobs -p | grep -v "^${ms}$") 2>/dev/null
echo "=== ${tag} node_run done $(date +%T)"
