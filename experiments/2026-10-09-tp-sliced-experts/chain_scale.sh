#!/usr/bin/env bash
# conc_scale arms (scale_arm.sh), ep and sl alternating on one fresh hold, from
# the frozen worktree tp-sliced-scale (7df917dd4d with the max_num_seqs <= 8
# guard lifted to 64):   chain_scale.sh <first: ep|sl> <n arms> <reserve GB>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees/tp-sliced-scale
first=$1 n=$2 res=$3
j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  if [[ ${arm} == ep ]]; then
    cb=${C}/vllm-cache-glm53-sc-ep-${j}; src=${C}/vllm-cache-glm53-ep-base; extra=""
  else
    cb=${C}/vllm-cache-glm53-sc-sl-${j}; src=${C}/vllm-cache-glm53-sliced
    extra="TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  mkdir -p ${cb}; cp -an ${src}/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  tag=scale-mtp3-r${res}-${arm}-${j}-${i}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/scale_arm.sh ${tag} RESERVE_GB=${res} PYTHONPATH=${WT} SERVE_CACHE_ROOT=${cb} ${extra}" \
    > ${D}/logs/serve/${tag}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
