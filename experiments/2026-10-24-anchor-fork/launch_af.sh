#!/usr/bin/env bash
# Anchor indexer fork/join arms on one hold, in order, from the frozen tree
# worktrees/${WT:-afork2} (branches anchor-indexer-fork{,-v2}) at the c=4 launcher config with
# the glue on, one cache copy per hold:
#   a = VLLM_ANCHOR_INDEXER_FORK=0   f = VLLM_ANCHOR_INDEXER_FORK=1
#   kinds: dec4 (nsys, 4 requests at ~5K / ~50K, node_run2.sh), probe (probe.py)
#     launch_af.sh <hold> <run tag> <kind>:<arm> ...
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-24-anchor-fork
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
G=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees WT=${WT:-afork2}
j=$1 run=$2; shift 2
P="PYTHONPATH=${W}/${WT} PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_MLA_QKV_GLUE=1"
a_env="${P} VLLM_ANCHOR_INDEXER_FORK=0"
f_env="${P} VLLM_ANCHOR_INDEXER_FORK=1"
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do
  [[ -z $(squeue -h -j "${j}" -o %T) ]] && { echo "hold ${j} left the queue"; exit 1; }
  sleep 20
done
sleep 5
cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
echo "$(date +%T) hold ${j} on $(squeue -h -j ${j} -o %N), ${WT} $(git -C ${W}/${WT} rev-parse --short HEAD)"
for spec in "$@"; do
  kind=${spec%%:*} arm=${spec##*:} envv=${spec##*:}_env tag=${run}${spec##*:}-${spec%%:*}-${j}
  case ${kind} in
    dec4) HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} dec4 ${cb} ${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1 ;;
    probe) HOLD_JOB=${j} ${E}/onnode.sh "${G}/probe_arm.sh ${tag} 4 4 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false ${!envv}" > ${X}/logs/${tag}.log 2>&1 ;;
  esac
  echo "$(date +%T) ${tag}: $(grep -a 'acc greedy\|agg\|=== \|server failed' ${X}/logs/${tag}.log | tail -3 | tr '\n' ' ')"
done
rm -rf ${cb} ${tc}
echo "=== launch_af ${run} done"
