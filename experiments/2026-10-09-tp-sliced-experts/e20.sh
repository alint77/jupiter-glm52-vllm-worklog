#!/usr/bin/env bash
# v29: fused prep, separate finalize writing the final output; v27 vs v29 x2
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v27 td_v29 >/dev/null 2>&1
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
ck() { run $1 timeout 1200 $PY kdev.py check --v "$2" --reps $3 2>&1 | grep "^{" | tail -2 | sed "s|^|gpu$1 $2: |"; }
b() { for v in td_v27 td_v29 td_v27 td_v29; do run $1 $PY kdev.py bench --v $v --m $2 --numa-node $1 --shared 1 --cells ${@:3} 2>/dev/null | grep "^{" | cut -c1-90; done; }
ck 0 td_v29 20 & ck 1 td_v29 20 &
b 2 8 38,0 38,4 50,4 & b 3 32 110,0 110,12 & wait
