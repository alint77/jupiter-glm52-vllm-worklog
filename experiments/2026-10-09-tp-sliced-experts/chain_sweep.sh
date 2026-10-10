#!/usr/bin/env bash
# MTP3 interactivity sweep (../2026-10-08-m32 chart config: c=8, 1.6M KV pool,
# n = 1, 2, 4, 8 in flight at 5K and 50K; reserve 3.6: EP at 7df917dd4d no
# longer boots at the chart's 3.0) or profiled windows at the same
# config, prod EP vs tp_sliced from the frozen worktree tp-sliced-b (vLLM
# 7df917dd4d), arms alternating on one fresh hold:
#   chain_sweep.sh <sweep|prof> <first arm: ep|sl> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
M=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees/tp-sliced-b
mode=$1 first=$2 n=$3
j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  if [[ ${arm} == ep ]]; then
    cb=${C}/vllm-cache-glm53-sw-ep-${j}; src=${C}/vllm-cache-glm53-ep-base; extra=""
  else
    cb=${C}/vllm-cache-glm53-sw-sl-${j}; src=${C}/vllm-cache-glm53-sliced
    extra="TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  mkdir -p ${cb}; cp -an ${src}/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  envs="PYTHONPATH=${WT} SERVE_CACHE_ROOT=${cb} ${extra}"
  tag=mtp3-c8-pool1600k-${arm}-${j}-${i}
  if [[ ${mode} == sweep ]]; then
    cmd="${M}/sweep_arm.sh ${tag} mtp 8 VLLM_TIERED_MOE_KV_POOL_SEQS=4 RESERVE_GB=${RESERVE_GB:-3.6} ${envs}"
  else
    cmd="${D}/prof_c8_arm.sh ${tag} ${envs}"
  fi
  HOLD_JOB=${j} ${E}/onnode.sh "${cmd}" > ${D}/logs/serve/${mode}-${tag}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
