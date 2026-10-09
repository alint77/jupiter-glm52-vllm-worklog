#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
# cold-only cells: every CTA works the cold tier -> the real kernel's C2C ceiling
./ab.sh e5-8 8 "0,8 0,12 8,8" "td_v20" "td_v20:TD_COMPUTE_ONLY"
./ab.sh e5-32 32 "0,12 0,16 12,12" "td_v20" "td_v20:TD_COMPUTE_ONLY"
ck() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 timeout 1200 $PY kdev.py check --v "$2" --reps $3 2>&1 | tail -1 | sed "s|^|gpu$1 $2 reps=$3: |"; }
for g in 0 1 2 3; do ck $g "td_v20:TD_NO_FLUSH_FENCE" 25 & done; wait
