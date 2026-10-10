#!/usr/bin/env bash
# Fresh ablation breakdown of v31 (= in-tree): full / no routed math / compute-only
# / compute-only no math, shared on and off; plus traced timelines at 38/4.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V=(td_v31 "td_v31:TD_ABL_NOMMA" "td_v31:TD_COMPUTE_ONLY" "td_v31:TD_COMPUTE_ONLY TD_ABL_NOMMA")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
b() { g=$1 m=$2 sh=$3; shift 3; for v in "${V[@]}"; do for rep in 1 2; do run $g $PY kdev.py bench --v "$v" --m $m --numa-node $g --shared $sh --cells "$@" 2>/dev/null | grep "^{" | sed "s|^|sh$sh |" | cut -c1-110; done; done; }
b 0 8 1 38,0 38,4 50,4 > logs/e26-g0.txt & b 1 32 1 110,0 110,12 > logs/e26-g1.txt &
b 2 8 0 38,0 38,4 50,4 > logs/e26-g2.txt & b 3 32 0 110,0 110,12 > logs/e26-g3.txt & wait
cat logs/e26-g*.txt
./tr.sh 8 38,4 td_v31 "td_v31:TD_COMPUTE_ONLY" 2>&1 | tail -40
