#!/usr/bin/env bash
# Same-node served A/B of Phase 1 on one fresh hold, arms in the given order:
#   base  worktrees/dcp-new  (4524b36e4e, prod FlashMLA library)
#   new   worktrees/fmla-p1  (80bd3cce5c, register-split FlashMLA library)
# c16 config (p1_arm.sh, reserve 4.0), conc_scale at n = 2, 8, 16 for 5K and
# 50K context, 2 reps, then GSM8K (200). Caches start from the c16 run 2264992.
#   NS=1,4,12 ab_launch_ns.sh base new   (n list from NS, no GSM8K)
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-11-flashmla-phase1
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
declare -A tree=([base]=dcp-new [new]=fmla-p1)
j=${HOLD:-$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")}
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for arm in "$@"; do
  cb=${C}/vllm-cache-glm53-p1-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tag=p1-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${P}/p1_arm.sh ${tag} RESERVE_GB=4.0 SCALE_NS=${NS} SCALE_NS50=${NS} REPS=2 PYTHONPATH=${W}/${tree[${arm}]} SERVE_CACHE_ROOT=${cb}" > ${P}/logs/${tag}.log 2>&1
done
scancel "${j}"
