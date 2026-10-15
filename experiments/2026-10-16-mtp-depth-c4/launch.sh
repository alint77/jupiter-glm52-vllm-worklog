#!/usr/bin/env bash
# MTP3 vs MTP5 at the c4 launcher config, one fresh hold per arm, worktree pf5
# (vLLM 144f2f4aee-equivalent), FCFS: the 600 s agentic c=4 timeline with
# outputs saved (timeline_save.py). Same capture list in both arms (adds the
# MTP5 verify sizes 18 and 24). ARMS="k3" runs a subset.
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-16-mtp-depth-c4
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf5
CS=1,2,3,4,5,6,7,8,12,16,18,24,64,128,256,384,512,640,768,896,1024
for arm in ${ARMS:-k3 k5}; do (
  j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
  echo "${arm} hold ${j}"
  cb=${C}/vllm-cache-glm53-mtpd-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=/e/fscratch/profound/${USER}/caches/triton-mtpd-${arm}-${j}; mkdir -p ${tc}
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=mtpd-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${P}/arm.sh ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} SPEC_K=${arm#k} CAPTURE_SIZES=${CS} SCRIPT=${P}/timeline_save.py TIMELINE_NOPROF=1 TIMELINE_DURATION=600 VLLM_SCHED_INTERLEAVE_SRPT=0" > ${P}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
