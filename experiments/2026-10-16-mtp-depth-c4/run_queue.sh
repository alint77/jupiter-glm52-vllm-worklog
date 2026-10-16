#!/usr/bin/env bash
# Work queue over standby holds: each hold, once RUNNING, claims queued arms
# (queue/<item> -> claimed/, atomic mv) and runs them one after another while
# it has >= 30 min left. Holds are never cancelled here: an idle hold stays
# available for more work until its time limit.
#   run_queue.sh <hold id>...     items: queue/k3, queue/k5, queue/k5-r2, queue/k3-r2
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-16-mtp-depth-c4
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf5
CS=1,2,3,4,5,6,7,8,12,16,18,24,64,128,256,384,512,640,768,896,1024
sq() {  # squeue -h -j $1 -o $2, retried: slurmctld timeouts print nothing
  local out i
  for i in 1 2 3 4 5; do
    out=$(squeue -h -j "$1" -o "$2" 2>/dev/null) && [[ -n ${out} ]] && { echo "${out}"; return; }
    [[ -n $(sacct -n -X -j "$1" -o State 2>/dev/null | grep -E "PENDING|RUNNING") ]] || return
    sleep 15
  done
}
left() {  # seconds left on hold $1
  local t; t=$(sq "$1" %L); [[ -z ${t} ]] && { echo 0; return; }
  local d=0; [[ ${t} == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
  IFS=: read -r a b c <<<"${t}"; [[ -z ${c} ]] && { c=${b}; b=${a}; a=0; }
  echo $(( d * 86400 + 10#${a} * 3600 + 10#${b} * 60 + 10#${c} ))
}
mkdir -p ${P}/queue/claimed ${P}/logs
for j in "$@"; do (
  until [[ $(sq "${j}" %T) == RUNNING ]]; do
    [[ -z $(sq "${j}" %T) ]] && { echo "$(date +%T) hold ${j} left the queue"; exit 0; }; sleep 20
  done
  echo "$(date +%T) hold ${j} RUNNING on $(sq ${j} %N)"
  sleep 5
  while (( $(left ${j}) >= 1800 )); do
    item=$(ls ${P}/queue | grep -v claimed | sort | head -1)
    [[ -z ${item} ]] && break
    mv ${P}/queue/${item} ${P}/queue/claimed/${item}.${j} 2>/dev/null || continue
    k=${item%%-*}; k=${k#k}
    cb=${C}/vllm-cache-glm53-mtpd-${item}-${j}; mkdir -p ${cb}
    cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
    tc=/e/fscratch/profound/${USER}/caches/triton-mtpd-${item}-${j}; mkdir -p ${tc}
    tag=mtpd-${item}-${j}
    echo "$(date +%T) hold ${j}: ${tag} (SPEC_K=${k})"
    HOLD_JOB=${j} ${E}/onnode.sh "${P}/arm.sh ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} SPEC_K=${k} CAPTURE_SIZES=${CS} SCRIPT=${P}/timeline_save.py TIMELINE_NOPROF=1 TIMELINE_DURATION=600 VLLM_SCHED_INTERLEAVE_SRPT=0" > ${P}/logs/${tag}.log 2>&1
    echo "$(date +%T) hold ${j}: ${tag} finished ($(tail -1 ${P}/logs/${tag}.log))"
  done
  echo "$(date +%T) hold ${j} IDLE, $(( $(left ${j}) / 60 )) min left (not cancelled)"
) & done
wait
