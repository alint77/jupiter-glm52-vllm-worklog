#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "td_v25:TD_R0S=2" "td_v25:TD_R0S=3" "td_v25:TD_R0S=4" >/dev/null 2>&1
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 900 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|$2 reps=$3: |"; }
ck 0 "td_v25:TD_R0S=2" 10 & ck 1 "td_v25:TD_R0S=3" 10 & ck 2 "td_v25:TD_R0S=4" 10 & ck 3 "td_v25:TD_R0S=2" 10 & wait
vs=(); for r in 1 2 3 4; do vs+=("td_v25:TD_R0S=$r|--shared 1"); done
./ab.sh e10 8 "38,4 38,0" "${vs[@]}"
./ab.sh e10-32 32 "110,12 110,0" "${vs[@]}"
