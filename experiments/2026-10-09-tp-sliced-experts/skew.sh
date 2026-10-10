#!/usr/bin/env bash
# Lever 6: is one GPU's sliced MoE kernel slower in isolation? td_v27 on each
# GPU of the node, NUMA bound, alone and then all four at once.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
hostname
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
b() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 $PY kdev.py bench --v td_v27 --m 8 --shared 1 --cells 38,0 38,4 0,8 2>&1 | grep -vi warn | sed "s|^|$2 gpu$1: |"; }
for g in 0 1 2 3; do b $g alone; done
for g in 0 1 2 3; do b $g conc & done; wait
