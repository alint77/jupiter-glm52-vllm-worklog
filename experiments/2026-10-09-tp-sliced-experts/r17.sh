#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V=${V:-td_v17}
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py check --v $V --reps 10 2>&1 | tail -4
./ab.sh ${TAG:-v17} 8 "38,4 38,0" "td_v15|--shared 1" "$V|--shared 1" "td_v15:TD_COMPUTE_ONLY|--shared 1" "$V:TD_COMPUTE_ONLY|--shared 1"
./ab.sh ${TAG:-v17}-32 32 "110,12" "td_v15|--shared 1" "$V|--shared 1" "td_v15:TD_COMPUTE_ONLY|--shared 1" "$V:TD_COMPUTE_ONLY|--shared 1"
