#!/usr/bin/env bash
# served sliced_decode.cu (copied as td_v57: v56 @64 + producer smem proxy fence): kdev check, adversarial (seeds 41-44), pytest
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V="td_v57:TD_MAX_TOKENS=64"
$PY -c "import kdev, sys; kdev.build_many(sys.argv[1:])" "$V" >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py check --v "$V" --reps 3 2>&1 | tail -4
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$i --membind=$i $PY adv_check.py --v "$V" --reps 3 --seed $((i + 41)) > logs/adv-v57-s$((i + 41)).out 2>&1 &
done
wait
for i in 0 1 2 3; do echo "== seed $((i + 41))"; tail -3 logs/adv-v57-s$((i + 41)).out; done
cd /e/project1/profound/alint77/vllm
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/vllm-pytest
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 .venv/bin/python -m pytest tests/kernels/moe/test_tiered_decode_sliced.py -q 2>&1 | tail -15
echo E77_DONE
