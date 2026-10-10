#!/usr/bin/env bash
# v28 (route_prep and finalize folded into the layer kernel, routed scale
# applied in-kernel): check on GPUs 0-1, v27 vs v28 interleaved on GPUs 2-3
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v27 td_v28 >/dev/null 2>&1
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
ck() { run $1 timeout 1200 $PY kdev.py check --v "$2" --reps $3 2>&1 | grep "^{" | tail -3 | sed "s|^|gpu$1 $2: |"; }
b() { for r in 1 2; do for v in td_v27 td_v28; do run $1 $PY kdev.py bench --v $v --m $2 --numa-node $1 --shared 1 --cells ${@:3} 2>/dev/null | grep "^{" | sed "s|^|r$r |"; done; done; }
ck 0 td_v28 30 & ck 1 td_v28 30 &
b 2 8 38,0 38,4 30,2 50,4 & b 3 32 110,0 110,12 84,8 & wait
