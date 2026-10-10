#!/usr/bin/env bash
# v33 (epilogue warp): check, bench vs v32 interleaved, per-warp trace
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v32 td_v34 "td_v34:TD_CTA_TRACE TD_UNIT_TRACE" "td_v34:TD_COMPUTE_ONLY" "td_v32:TD_COMPUTE_ONLY" 2>&1 | grep -iE "error|warning: v" | head
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
ck() { run $1 timeout 1200 $PY kdev.py check --v "$2" --reps $3 2>&1 | grep "^{" | tail -1 | sed "s|^|gpu$1 $2: |"; }
b() { g=$1 m=$2; shift 2; for rep in 1 2; do for v in td_v32 td_v34 td_v32:TD_COMPUTE_ONLY td_v34:TD_COMPUTE_ONLY; do run $g $PY kdev.py bench --v "$v" --m $m --numa-node $g --shared 1 --cells "$@" 2>/dev/null | grep "^{"; done; done; }
ck 0 td_v34 20 > logs/e34-ck0.txt &
ck 1 td_v34 20 > logs/e34-ck1.txt &
b 2 8 38,0 38,4 50,4 > logs/e34-g2.txt & b 3 32 110,0 110,12 > logs/e34-g3.txt & wait
cat logs/e34-ck*.txt
$PY abl_sum.py logs/e34-g2.txt logs/e34-g3.txt
for mc in "8 38,4" "32 110,12"; do set -- $mc
  run 0 $PY kdev.py once --v "td_v34:TD_CTA_TRACE TD_UNIT_TRACE" --m $1 --cell $2 --n 4 --shared 1 --numa-node 0 --trace logs/u34-$1.pt >/dev/null 2>&1
  echo "## v33 M=$1 $2"; $PY warp_tl.py logs/u34-$1.pt
done
