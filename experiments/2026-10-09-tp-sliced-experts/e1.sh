#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v18f "td_v17:TD_NO_FLUSH_FENCE" "td_v18f:TD_NO_FLUSH_FENCE" >/dev/null 2>&1
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 900 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|$2 reps=$3: |"; }
ck 0 td_v18f 20 & ck 1 "td_v17:TD_NO_FLUSH_FENCE" 60 & ck 2 "td_v18f:TD_NO_FLUSH_FENCE" 30 & wait
./ab.sh e1 8 "38,4 38,0" "td_v17|--shared 1" "td_v18f|--shared 1" "td_v17:TD_NO_FLUSH_FENCE|--shared 1" "td_v18f:TD_NO_FLUSH_FENCE|--shared 1"
./ab.sh e1-32 32 "110,12 110,0" "td_v17|--shared 1" "td_v18f|--shared 1" "td_v17:TD_NO_FLUSH_FENCE|--shared 1" "td_v18f:TD_NO_FLUSH_FENCE|--shared 1"
