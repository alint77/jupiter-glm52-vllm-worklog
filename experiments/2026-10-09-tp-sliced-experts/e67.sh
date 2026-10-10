#!/usr/bin/env bash
# adversarial check of the shipped kernel source (td_v54), four seeds, one per GPU
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev; kdev.build_many(['td_v54'])"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$i --membind=$i $PY adv_check.py --v td_v54 --reps ${REPS:-5} --seed $((i + 11)) > logs/adv-td_v54-s$((i + 11)).out 2>&1 &
done
wait
for i in 0 1 2 3; do echo "== seed $((i + 11))"; tail -2 logs/adv-td_v54-s$((i + 11)).out; done
