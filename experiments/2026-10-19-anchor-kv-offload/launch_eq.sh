#!/usr/bin/env bash
# Verify-step cost of each KV placement at an equal hot set (c=4 launcher
# config, 0.8M pool, worktrees/ahc-s3 = 3ab9242898), every arm pinned to
# VLLM_TIERED_MOE_HOT_EXPERTS=${HOT:-12831} (what A fits):
#   x = A: all main MLA KV in HBM (MLA_CACHE_TIER=hbm)
#   y = B: skip layers' KV on Grace, staged per step (skip_host_uva)
#   z = C: B + anchors' KV on Grace behind the hot cache (prod)
# Same hot experts, so the MoE is identical; the differences are the KV
# placement. dec4 windows (4 decoders at ~5K / ~50K), one fresh hold per
# order, arms in that order on it.
#   launch_eq.sh ["<order>"...]   (default xyz yzx zxy)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/anchor-hot
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P="PYTHONPATH=${W}/ahc-s3 PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_TIERED_MOE_HOT_EXPERTS=${HOT:-12831}"
x_env="${P} MLA_CACHE_TIER=hbm VLLM_ANCHOR_HOT_CACHE=0"
y_env="${P} VLLM_ANCHOR_HOT_CACHE=0"
z_env="${P} VLLM_ANCHOR_HOT_CACHE=1"
orders=("$@"); (( ${#orders[@]} )) || orders=(xyz yzx zxy)
for order in "${orders[@]}"; do (
  j=$(sbatch --parsable --time=$(( ${#order} * 17 + 10 )) "${E}/hold.sbatch")
  echo "${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  # one cache copy per hold (arms run one after another on it), removed after:
  # ~20K files per copy, and per-arm copies filled the fscratch inode quota
  cb=${C}/vllm-hold-${j} tc=${C}/triton-hold-${j}; mkdir -p ${cb} ${tc}
  cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  for ((i = 0; i < ${#order}; i++)); do
    arm=${order:i:1} tag=ak${order:i:1}-dec4-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} dec4 ${cb} ${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== \|server failed' ${X}/logs/${tag}.log | tail -1)"
  done
  rm -rf ${cb} ${tc}
) & done
wait
echo "=== launch_eq done"
