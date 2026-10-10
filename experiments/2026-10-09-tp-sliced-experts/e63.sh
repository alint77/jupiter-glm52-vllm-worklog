#!/usr/bin/env bash
# v53 (Astra 12 defaults; NOAHEAD, RLIST) vs v48 / v51. e63.sh <8|32|trace>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v48" "td_v53" "td_v53:TD_NOAHEAD" "td_v53:TD_RLIST" "td_v53:TD_NOAHEAD TD_RLIST")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v53 "td_v53:TD_NOAHEAD" "td_v53:TD_RLIST" "td_v53:TD_NOAHEAD TD_RLIST" "td_v53:TD_NOAHEAD TD_RLIST TD_UNIT_TRACE TD_CTA_TRACE" "td_v53:TD_UNIT_TRACE TD_CTA_TRACE"
  for v in td_v53 "td_v53:TD_NOAHEAD" "td_v53:TD_RLIST" "td_v53:TD_NOAHEAD TD_RLIST"; do
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 60 2>&1 | tail -1
  done
  ./ab.sh e63 8 "38,0 38,4 50,4" "${vs[@]}"
elif [[ $1 == 32 ]]; then ./ab.sh e63-16 16 "70,6" "${vs[@]}"; ./ab.sh e63-32 32 "110,0 110,12" "${vs[@]}"
else
  for v in "td_v53:TD_UNIT_TRACE TD_CTA_TRACE" "td_v53:TD_NOAHEAD TD_RLIST TD_UNIT_TRACE TD_CTA_TRACE"; do
  for mc in "8 38,4" "8 38,0" "32 110,12"; do set -- $mc
    o=logs/u63-$1-${2/,/_}-$(echo "$v" | grep -q RLIST && echo nr || echo b).pt
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace $o >/dev/null 2>&1
    echo "######## $v M=$1 $2"; $PY trace_bd.py $o 4 8 | head -24; $PY ent_tl.py $o 0 | head -3; $PY ent_tl.py $o 0 | tail -1; $PY sched_tl.py $o
  done; done
fi
