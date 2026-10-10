#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 timeout 900 $PY kdev.py check --v td_v28 --reps 2 2>&1 | grep "^{" | sort | uniq -c | sort -k2 | head -40
