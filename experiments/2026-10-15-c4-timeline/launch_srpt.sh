#!/usr/bin/env bash
# SRPT A/B: the c=4 agentic timeline (timeline_run.py, 600 s, no profiler) on
# pf3 (877df9b262) with VLLM_SCHED_INTERLEAVE_SRPT=0 (FCFS) vs 1, one fresh
# hold each, own caches.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-15-c4-timeline
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf3
for s in 0 1; do (
  j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-srpt${s}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=/e/fscratch/profound/${USER}/caches/triton-srpt${s}-${j}; mkdir -p ${tc}
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=srpt${s}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} SCRIPT=${D}/timeline_run.py TIMELINE_NOPROF=1 TIMELINE_DURATION=600 VLLM_SCHED_INTERLEAVE_SRPT=${s}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
