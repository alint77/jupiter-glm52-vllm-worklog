#!/usr/bin/env bash
# Profile windows for the 4-token verify step: DFlash2 k=3 on 2110228, MTP3 on 2110229.
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
for pair in "2110228 dflash2 df2k3-prof" "2110229 mtp mtpk3-prof"; do
  set -- $pair
  (
    until [[ "$(squeue -h -j $1 -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
    SPEC=$2 SPEC_K=3 PROF_TAG=$3 bash $B/launch_prof.sh glm $1 --limit-requests 60
  ) &
done
wait
