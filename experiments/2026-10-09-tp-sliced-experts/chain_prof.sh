#!/usr/bin/env bash
# Profiled MTP3 arms (prof_c8_arm.sh: torch-profiler windows + allocator
# snapshots) at the interactivity-chart config, prod EP vs tp_sliced from the
# frozen worktree tp-sliced-mem (vLLM 7df917dd4d + the local MEM_SNAPSHOT_DIR
# probe in gpu_worker.py), arms alternating on one fresh hold:
#   chain_prof.sh <first arm: ep|sl> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees/tp-sliced-mem
first=$1 n=$2
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  if [[ ${arm} == ep ]]; then
    cb=${C}/vllm-cache-glm53-pm-ep-${j}; src=${C}/vllm-cache-glm53-ep-base; extra=""
  else
    cb=${C}/vllm-cache-glm53-pm-sl-${j}; src=${C}/vllm-cache-glm53-sliced
    extra="TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  mkdir -p ${cb}; cp -an ${src}/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  tag=pm-mtp3-c8-${arm}-${j}-${i}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/prof_c8_arm.sh ${tag} PYTHONPATH=${WT} SERVE_CACHE_ROOT=${cb} ${extra}" \
    > ${D}/logs/serve/prof-${tag}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
