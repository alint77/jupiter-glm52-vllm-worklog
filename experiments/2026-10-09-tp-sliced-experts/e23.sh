#!/usr/bin/env bash
# v30 (fast prep, end-of-kernel conversion): check, v27 vs v30 x2, phase probe
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v27 td_v30 td_v30:TD_PROBE 2>&1 | grep -i error
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
ck() { run $1 timeout 1200 $PY kdev.py check --v "$2" --reps $3 2>&1 | grep "^{" | tail -2 | sed "s|^|gpu$1 $2: |"; }
b() { for v in td_v27 td_v30 td_v27 td_v30; do run $1 $PY kdev.py bench --v $v --m $2 --numa-node $1 --shared 1 --cells ${@:3} 2>/dev/null | grep "^{" | cut -c1-90; done; }
pr() { run $1 $PY kdev.py once --v td_v30:TD_PROBE --m $2 --cell $3 --n 3 --shared 1 --numa-node $1 --trace logs/p30-$2.pt >/dev/null 2>&1; echo "probe m=$2 $3: $($PY probe30.py logs/p30-$2.pt)"; }
ck 0 td_v30 20 & ck 1 td_v30 20 &
b 2 8 38,0 38,4 50,4 & b 3 32 110,0 110,12 & wait
pr 0 8 38,4 & pr 1 32 110,0 & wait
