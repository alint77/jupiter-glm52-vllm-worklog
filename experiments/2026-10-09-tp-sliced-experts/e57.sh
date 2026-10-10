#!/usr/bin/env bash
# v46 finisher warp (w13 handoff off consumer warp 0) vs v45. e57.sh <8|32|trace>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v45" "td_v46" "td_v46:TD_NO_SREADY")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v46 "td_v46:TD_NO_SREADY" "td_v46:TD_UNIT_TRACE TD_CTA_TRACE"
  for v in td_v46 "td_v46:TD_NO_SREADY"; do
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 30 2>&1 | tail -1
  done
  ./ab.sh e57 8 "38,0 38,4 50,4" "${vs[@]}"
elif [[ $1 == 32 ]]; then ./ab.sh e57-16 16 "70,6" "${vs[@]}"; ./ab.sh e57-32 32 "110,0 110,12" "${vs[@]}"
else
  v="td_v46:TD_UNIT_TRACE TD_CTA_TRACE"
  for mc in "8 38,4" "32 110,12"; do set -- $mc
    o=logs/u57-$1.pt
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace $o >/dev/null 2>&1
    echo "######## $v M=$1 $2"; $PY trace_bd.py $o 4 8 | head -24; $PY hand_tl.py $o; $PY prod_tl.py $o 2>/dev/null | tail -9; $PY sched_tl.py $o
  done
fi
