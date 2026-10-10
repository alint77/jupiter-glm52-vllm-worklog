#!/usr/bin/env bash
# v37 switches (Astra consult 9): check + benches. e45.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
A="TD_MMA_NV TD_MB2 TD_GLOOP TD_ACQREL"
vs=("td_v32" "td_v37:TD_MMA_NV" "td_v37:TD_MB2" "td_v37:TD_GLOOP" "td_v37:TD_ACQREL" "td_v37:$A")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  for v in "td_v37:$A" "td_v37:TD_GLOOP" "td_v37:TD_ACQREL"; do
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 20 2>&1 | tail -1; done
  ./ab.sh e45 8 "38,0 38,4 50,4" "${vs[@]}"
else ./ab.sh e45-32 32 "110,0 110,12" "${vs[@]}"; fi
