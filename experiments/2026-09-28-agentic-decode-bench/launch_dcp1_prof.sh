#!/usr/bin/env bash
# Profile windows only, GLM DCP1 250K (the timing run is rows-glm-dcp1-250k).
cd /e/project1/profound/alint77/vllm
export DCP=1 MAX_MODEL_LEN=250000 VLLM_TIERED_MOE_RELAX_SHAPE=1 PREFIX_CACHING=1
until [[ "$(squeue -h -j 2106021 -o %T)" == RUNNING ]]; do sleep 20; done; sleep 15
PROF_TAG=dcp1-prof agent_space/experiments/2026-09-28-agentic-decode-bench/launch_prof.sh glm 2106021 --limit-requests 60
wait
