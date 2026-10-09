#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 900 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|$2 reps=$3: |"; }
ck 0 td_v22 15 & ck 1 "td_v22:TD_NO_FLUSH_FENCE" 15 & ck 2 td_v22 15 & ck 3 "td_v22:TD_NO_FLUSH_FENCE" 15 & wait
./ab.sh e6 8 "38,4 38,0" "td_v20|--shared 1" "td_v20:TD_NO_FLUSH_FENCE|--shared 1" "td_v22|--shared 1" "td_v22:TD_NO_FLUSH_FENCE|--shared 1"
./ab.sh e6-32 32 "110,12 110,0" "td_v20|--shared 1" "td_v20:TD_NO_FLUSH_FENCE|--shared 1" "td_v22|--shared 1" "td_v22:TD_NO_FLUSH_FENCE|--shared 1"
