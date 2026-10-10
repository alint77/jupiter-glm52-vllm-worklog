#!/usr/bin/env bash
# reproduce adv seed 32's one failure (m=47 tiny nohot pad shared, rep 2): v56@64 seed 32 twice,
# v56@64 seed 35, td_v55 seed 32 (control, m <= 32)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V64="td_v56:TD_MAX_TOKENS=64"
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 $PY adv_check.py --v "$2" --reps ${REPS:-3} --seed $3 > logs/e74-g$1.out 2>&1; }
run 0 "$V64" 32 & run 1 "$V64" 32 & run 2 "$V64" 35 & run 3 td_v55 32 &
wait
for g in 0 1 2 3; do echo "== g$g"; grep -v "^ptxas\|^built" logs/e74-g$g.out | cut -c1-260 | tail -4; done
echo E74_DONE
