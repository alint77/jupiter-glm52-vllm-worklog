#!/usr/bin/env bash
# KV pool size probes on the shipped tree (worktrees/ahc-s2 = e51272479d's
# tree, VLLM_ANCHOR_HOT_CACHE=1), dec4 at the c=4 launcher config with
# POOL_SEQS full-length (400K) sequences in the pool, one fresh hold each:
# planner residency, free HBM and Grace after startup, and the dec4 step.
#   launch_pool.sh <pool_seqs>...
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/anchor-hot
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
for p in "$@"; do (
  j=$(sbatch --parsable --time=45 "${E}/hold.sbatch")
  echo "pool ${p} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=akp${p}-dec4-${j}
  cb=${C}/vllm-${tag} tc=${C}/triton-${tag}
  mkdir -p ${cb} ${tc}
  cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} dec4 ${cb} ${tc} PYTHONPATH=${W}/ahc-s2 PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_ANCHOR_HOT_CACHE=1 POOL_SEQS=${p}" > ${X}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(grep -a '=== \|server failed' ${X}/logs/${tag}.log | tail -1)"
) & done
wait
echo "=== launch_pool done"
