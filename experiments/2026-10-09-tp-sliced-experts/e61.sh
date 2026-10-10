#!/usr/bin/env bash
# v51 finisher done path (late y13 zeroing, acq_rel fences, one-fence release,
# acq_rel count RMW) vs v48. e61.sh <8|32|trace>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v48" "td_v51" "td_v51:TD_ONEREL" "td_v51:TD_ATOM_AR" "td_v51:TD_ONEREL TD_ATOM_AR")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v51 "td_v51:TD_ONEREL" "td_v51:TD_ATOM_AR" "td_v51:TD_ONEREL TD_ATOM_AR" "td_v51:TD_UNIT_TRACE TD_CTA_TRACE" "td_v51:TD_ONEREL TD_ATOM_AR TD_UNIT_TRACE TD_CTA_TRACE"
  for v in td_v51 "td_v51:TD_ONEREL" "td_v51:TD_ATOM_AR" "td_v51:TD_ONEREL TD_ATOM_AR"; do
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 30 2>&1 | tail -1
  done
  ./ab.sh e61 8 "38,0 38,4 50,4" "${vs[@]}"
elif [[ $1 == 32 ]]; then ./ab.sh e61-16 16 "70,6" "${vs[@]}"; ./ab.sh e61-32 32 "110,0 110,12" "${vs[@]}"
else
  for v in "td_v51:TD_UNIT_TRACE TD_CTA_TRACE" "td_v51:TD_ONEREL TD_ATOM_AR TD_UNIT_TRACE TD_CTA_TRACE"; do
  for mc in "8 38,4" "8 38,0" "32 110,12"; do set -- $mc
    o=logs/u61-$1-${2/,/_}-$(echo "$v" | grep -q ONEREL && echo oa || echo b).pt
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace $o >/dev/null 2>&1
    echo "######## $v M=$1 $2"; $PY trace_bd.py $o 4 8 | head -24; $PY hand_tl.py $o | head -9; $PY prod_tl.py $o 2>/dev/null | tail -9 | head -5; $PY fin_tl.py $o
  done; done
fi
