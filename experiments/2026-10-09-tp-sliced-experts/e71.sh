#!/usr/bin/env bash
# td_v56 (TD_MAX_TOKENS 32 / 64): ptxas, kdev check, adversarial check (seeds 31-34, one per GPU)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V64="td_v56:TD_MAX_TOKENS=64"
./kb.sh td_v55 td_v56 "$V64"
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py check --v "$V64" --reps 3 2>&1 | tail -4
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$i --membind=$i $PY adv_check.py --v "$V64" --reps ${REPS:-3} --seed $((i + 31)) > logs/adv-v56-64-s$((i + 31)).out 2>&1 &
done
wait
for i in 0 1 2 3; do echo "== seed $((i + 31))"; tail -3 logs/adv-v56-64-s$((i + 31)).out; done
echo E71_DONE
