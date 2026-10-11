#!/usr/bin/env bash
# Same-node served A/B of the MTP layer's KV in HBM (vllm c047194130) on one
# fresh hold, arms in the given order:
#   base  worktrees/fmla-p1  (80bd3cce5c: MTP layer KV on Grace)
#   new   worktrees/mtpkv    (c047194130: MTP layer KV in HBM, as planned)
# c16 config (../2026-10-11-flashmla-phase1/p1_arm.sh, reserve 4.0), conc_scale
# at n = 4, 8, 16 for 5K and 50K context, 2 reps. No GSM8K: only the KV
# placement changes (memory: no-gsm8k-in-serving-ab).
#   ab_launch.sh base new      (and new base on a second node)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-11-mtp-kv-hbm
P=agent_space/experiments/2026-10-11-flashmla-phase1
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
declare -A tree=([base]=fmla-p1 [new]=mtpkv)
j=${HOLD:-$(sbatch --parsable --time=02:00:00 "${E}/hold.sbatch")}
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for arm in "$@"; do
  cb=${C}/vllm-cache-glm53-mkv-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tag=mkv-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${P}/p1_arm.sh ${tag} RESERVE_GB=4.0 SCALE_NS=4,8,16 SCALE_NS50=4,8,16 REPS=2 PYTHONPATH=${W}/${tree[${arm}]} SERVE_CACHE_ROOT=${cb}" > ${X}/logs/${tag}.log 2>&1
done
scancel "${j}"
