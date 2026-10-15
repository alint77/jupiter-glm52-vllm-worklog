#!/usr/bin/env bash
# Served check of the inline-system merge fix (vLLM 144f2f4aee, worktree pf5,
# = pf3 + the fix), one fresh hold per arm:
#   pfx:  prefix_repro.py at c=1 (expect #113->#114 ~240K hit, #16->#17 ~68K);
#   fcfs: c=4 timeline, 600 s, no profiler (vs srpt0-2334591 on pf3);
#   srpt: same with VLLM_SCHED_INTERLEAVE_SRPT=1 (vs srpt1-2334590).
# ARMS="pfx fcfs" runs a subset.
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-15-prefix-miss
D=agent_space/experiments/2026-10-15-c4-timeline
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf5
mkdir -p ${P}/logs
for arm in ${ARMS:-pfx fcfs srpt}; do (
  j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-mfix-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=/e/fscratch/profound/${USER}/caches/triton-mfix-${arm}-${j}; mkdir -p ${tc}
  case ${arm} in
    pfx) extra="SCRIPT=${D}/prefix_repro.py" ;;
    fcfs) extra="SCRIPT=${D}/timeline_run.py TIMELINE_NOPROF=1 TIMELINE_DURATION=600 VLLM_SCHED_INTERLEAVE_SRPT=0" ;;
    srpt) extra="SCRIPT=${D}/timeline_run.py TIMELINE_NOPROF=1 TIMELINE_DURATION=600 VLLM_SCHED_INTERLEAVE_SRPT=1" ;;
  esac
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=mfix-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} ${extra}" > ${P}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
