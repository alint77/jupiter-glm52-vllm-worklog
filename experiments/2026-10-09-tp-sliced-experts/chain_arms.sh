#!/usr/bin/env bash
# Served arms back to back on one fresh hold with the standard harness
# (../2026-10-07-mem-reclaim/arm.sh), each from its own frozen worktree.
#   chain_arms.sh <run|prof> <arm>...    arms: ep, sla (tp-sliced-a), slb (tp-sliced-b)
# run: arm.sh (GSM8K 200, agentic task set, long-context decode, TTFT)
# prof: prof_arm.sh (c=1 5K/50K windows under the torch profiler)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees
mode=$1; shift
j=$(sbatch --parsable --time=01:40:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
i=0
for arm in "$@"; do
  i=$((i + 1))
  case ${arm} in
    ep) envs="PYTHONPATH=${WT}/ep-base SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-ep-base" ;;
    sla) envs="PYTHONPATH=${WT}/tp-sliced-a SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-sliced TIERED_MOE_LAYOUT=tp_sliced REPLICAS=" ;;
    slb)
      # its own compile cache: a graph traced before the runner released its
      # epilogue must not be reused
      cb=${C}/vllm-cache-glm53-sliced-b-${j}
      mkdir -p ${cb}; cp -an ${C}/vllm-cache-glm53-sliced/{modelinfos,torch_extensions} ${cb}/
      envs="PYTHONPATH=${WT}/tp-sliced-b SERVE_CACHE_ROOT=${cb} TIERED_MOE_LAYOUT=tp_sliced REPLICAS=" ;;
  esac
  if [[ ${mode} == prof ]]; then
    cmd="${envs} ARM_ENVS=1 ${D}/prof_arm.sh ${arm} ${arm}-${j}-${i}"
  else
    cmd="${envs} ${A} ${arm}-${j}-${i}"
  fi
  HOLD_JOB=${j} ${E}/onnode.sh "${cmd}" > ${D}/logs/serve/${mode}-${arm}-${j}-${i}.log 2>&1
done
scancel "${j}"
