#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v18 td_v19 "td_v19:TD_NO_FLUSH_FENCE" 2>&1 | grep -i error | head
for v in td_v18 td_v19 "td_v19:TD_NO_FLUSH_FENCE"; do
  ( g=$((n++)); ) ; done
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py check --v td_v18 --reps 10 2>&1 | tail -1 &
CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py check --v td_v19 --reps 10 2>&1 | tail -1 &
CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=2 --membind=2 $PY kdev.py check --v "td_v19:TD_NO_FLUSH_FENCE" --reps 10 2>&1 | tail -1 &
wait
./ab.sh v19 8 "38,4 38,0" "td_v17|--shared 1" "td_v18|--shared 1" "td_v19|--shared 1" "td_v19:TD_NO_FLUSH_FENCE|--shared 1"
./ab.sh v19-32 32 "110,12" "td_v17|--shared 1" "td_v18|--shared 1" "td_v19|--shared 1" "td_v19:TD_NO_FLUSH_FENCE|--shared 1"
