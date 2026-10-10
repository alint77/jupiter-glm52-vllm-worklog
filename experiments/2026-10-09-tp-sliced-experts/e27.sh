#!/usr/bin/env bash
# v32 ablation grid: where the consumer's per-unit time goes (compute-only and
# with loads), M=8 and M=32, shared expert on
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
C=TD_COMPUTE_ONLY
V=(td_v32 "td_v32:TD_ABL_NOMMA TD_ABL_NOFLUSH TD_ABL_NOSHMMA" "td_v32:$C" "td_v32:$C TD_ABL_NOMMA"
   "td_v32:$C TD_ABL_NODECODE" "td_v32:$C TD_ABL_NOFLUSH" "td_v32:$C TD_ABL_NOMMA TD_ABL_NOFLUSH"
   "td_v32:$C TD_ABL_NOSHMMA" "td_v32:$C TD_ABL_NOMMA TD_ABL_NOFLUSH TD_ABL_NOSHMMA")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
b() { g=$1 m=$2; shift 2; for v in "${V[@]}"; do for rep in 1 2; do run $g $PY kdev.py bench --v "$v" --m $m --numa-node $g --shared 1 --cells "$@" 2>/dev/null | grep "^{"; done; done; }
b 0 8 38,0 38,4 > logs/e27-g0.txt & b 1 32 110,0 110,12 > logs/e27-g1.txt &
b 2 8 38,0 38,4 > logs/e27-g2.txt & b 3 32 110,0 110,12 > logs/e27-g3.txt & wait
/e/project1/profound/alint77/vllm/.venv/bin/python abl_sum.py logs/e27-g0.txt logs/e27-g1.txt; echo; /e/project1/profound/alint77/vllm/.venv/bin/python abl_sum.py logs/e27-g2.txt logs/e27-g3.txt
