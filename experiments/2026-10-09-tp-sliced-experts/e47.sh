#!/usr/bin/env bash
# v37 combined + w2 claims of GR1 units; with / without ACQREL. e47.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
B="TD_MMA_NV TD_MB2 TD_GLOOP"
vs=("td_v32" "td_v37:$B" "td_v37:$B TD_ACQREL" "td_v37:$B TD_GR1=2" "td_v37:$B TD_GR1=4" "td_v37:$B TD_GR1=4 TD_ACQREL")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "td_v37:$B TD_GR1=4" --reps 10 2>&1 | tail -1
  ./ab.sh e47 8 "38,0 38,4 50,4" "${vs[@]}"
else ./ab.sh e47-32 32 "110,0 110,12" "${vs[@]}"; fi
