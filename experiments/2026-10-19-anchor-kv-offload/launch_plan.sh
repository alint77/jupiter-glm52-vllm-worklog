#!/usr/bin/env bash
# Planner probes for the per-token HBM cost of each KV placement (c=4
# config, shipped tree ahc-s2): A = all main KV in HBM (MLA_CACHE_TIER=hbm),
# B = skip layers on Grace (skip_host_uva, no hot cache), each at 0.8M and
# 1.6M pools (POOLS="2 4"), one hold per config, sequential on it.
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
W=/e/fscratch/profound/${USER}/worktrees
C=/e/fscratch/profound/${USER}/caches/anchor-hot
P="PYTHONPATH=${W}/ahc-s2 PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_ANCHOR_HOT_CACHE=0"
declare -A cfg=([A]="${P} MLA_CACHE_TIER=hbm" [B]="${P}")
for k in A B; do (
  j=$(sbatch --parsable --time=20 "${E}/hold.sbatch")
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 15; done
  sleep 5
  for p in ${POOLS:-2 4}; do
    tag=plan${k}${p}-${j}
    HOLD_JOB=${j} ${E}/onnode.sh "${X}/probe_plan.sh ${tag} ${cfg[${k}]} POOL_SEQS=${p} SERVE_CACHE_ROOT=${C}/vllm-${tag} TRITON_CACHE_DIR=${C}/triton-${tag}" 2>&1 | tail -1
  done
) & done
wait
echo "=== launch_plan done"
