#!/usr/bin/env bash
# L2 prefetch of later units in multi-unit groups (v36 TD_L2PF)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "td_v36:TD_L2PF=4" --reps 5 2>&1 | tail -3
vs=("td_v32|--shared 1" "td_v36:TD_L2PF=2|--shared 1" "td_v36:TD_L2PF=4|--shared 1" "td_v36:TD_L2PF=8|--shared 1")
./ab.sh e38 8 "38,0 38,4 50,4" "${vs[@]}"
./ab.sh e38-32 32 "110,0 110,12" "${vs[@]}"
