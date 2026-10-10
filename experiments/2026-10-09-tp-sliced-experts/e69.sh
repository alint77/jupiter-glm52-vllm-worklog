#!/usr/bin/env bash
# adversarial check round 2 (Astra review 13 cases): e69.sh <variant>, seeds 21-24, one per GPU
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
v=$1
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" $v
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$i --membind=$i $PY adv_check.py --v $v --reps ${REPS:-3} --seed $((i + 21)) > logs/adv2-$v-s$((i + 21)).out 2>&1 &
done
wait
for i in 0 1 2 3; do echo "== $v seed $((i + 21))"; tail -4 logs/adv2-$v-s$((i + 21)).out; done
