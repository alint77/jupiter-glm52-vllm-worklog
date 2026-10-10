#!/usr/bin/env bash
# Nsight Systems MTP3 arms (nsys_c8_arm.sh: per-node graph kernels, one report
# per window) at the interactivity-chart config, prod EP vs tp_sliced from the
# frozen worktree tp-sliced-b (vLLM 7df917dd4d), arms alternating on one hold:
#   chain_nsys.sh <first arm: ep|sl> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees/tp-sliced-b
first=$1 n=$2
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  if [[ ${arm} == ep ]]; then
    cb=${C}/vllm-cache-glm53-ns-ep-${j}; src=${C}/vllm-cache-glm53-ep-base; extra=""
  else
    cb=${C}/vllm-cache-glm53-ns-sl-${j}; src=${C}/vllm-cache-glm53-sliced
    extra="TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  mkdir -p ${cb}; cp -an ${src}/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  tag=ns-mtp3-c8-${arm}-${j}-${i}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/nsys_c8_arm.sh ${tag} PYTHONPATH=${WT} SERVE_CACHE_ROOT=${cb} ${extra}" \
    > ${D}/logs/serve/nsys-${tag}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
