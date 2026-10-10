#!/usr/bin/env bash
# Same-node served A/B of the DCP one-shot kernels: one fresh hold, the arms in
# the given order, each from its frozen worktree (worktrees/dcp-<arm>), on the
# c16 config (c16_arm.sh, reserve 4.0) with conc_scale at n = 1, 4, 8, 16.
#   ab_launch.sh base new      (and new base on a second node)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-dcp-oneshot-bw
S=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
j=$(sbatch --parsable --time=01:45:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for arm in "$@"; do
  cb=${C}/vllm-cache-glm53-dcp-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-sliced/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  tag=dcp-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${S}/c16_arm.sh ${tag} RESERVE_GB=4.0 SCALE_NS=1,4,8,16 SCALE_NS50=1,4,8,16 REPS=2 PYTHONPATH=${W}/dcp-${arm} SERVE_CACHE_ROOT=${cb}" > ${D}/logs/${tag}.log 2>&1
done
scancel "${j}"
