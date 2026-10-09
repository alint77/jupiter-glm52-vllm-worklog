#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v26 "td_v26:TD_R0S=2" "td_v26:TD_GR1=2" "td_v26:TD_R0S=2 TD_GR1=2" 2>&1 | grep -i error | head -5
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 900 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|$2 reps=$3: |"; }
ck 0 td_v26 15 & ck 1 "td_v26:TD_R0S=2" 15 & ck 2 "td_v26:TD_R0S=2 TD_GR1=2" 15 & ck 3 td_v26 15 & wait
vs=("td_v25|--shared 1" "td_v26|--shared 1" "td_v26:TD_R0S=2|--shared 1" "td_v26:TD_GR1=2|--shared 1" "td_v26:TD_R0S=2 TD_GR1=2|--shared 1")
./ab.sh e13 8 "38,4 38,0" "${vs[@]}"
./ab.sh e13-32 32 "110,12 110,0" "${vs[@]}"
