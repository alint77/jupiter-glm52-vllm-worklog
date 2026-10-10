#!/usr/bin/env bash
# v39 (per-call w2 claim size) vs v32, +/- ACQREL. e49.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v32" "td_v39" "td_v39:TD_ACQREL")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  for v in td_v39 "td_v39:TD_ACQREL"; do echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 30 2>&1 | tail -1; done
  ./ab.sh e49 8 "38,0 38,4 50,4" "${vs[@]}"
else ./ab.sh e49-16 16 "70,6" "${vs[@]}"; ./ab.sh e49-32 32 "110,0 110,12" "${vs[@]}"; fi
