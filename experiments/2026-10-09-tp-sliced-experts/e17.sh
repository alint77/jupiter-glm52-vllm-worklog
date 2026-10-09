#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v27 "td_v27:TD_GR1=2" >/dev/null 2>&1
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 1000 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|gpu$1 $2 reps=$3: |"; }
ck 0 td_v27 25 & ck 1 td_v27 25 & ck 2 "td_v27:TD_GR1=2" 25 & ck 3 "td_v27:TD_GR1=2" 25 & wait
