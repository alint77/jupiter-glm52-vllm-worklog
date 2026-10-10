#!/usr/bin/env bash
# v53 hot-only NOAHEAD (+/- RLIST) vs v48 / v53 variants. e65.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v48" "td_v53:TD_NOAHEAD" "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT" "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT TD_RLIST" "td_v53:TD_RLIST")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "td_v53:TD_NOAHEAD" "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT" "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT TD_RLIST" "td_v53:TD_RLIST"
  for v in "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT" "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT TD_RLIST"; do
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 30 2>&1 | tail -1
  done
  ./ab.sh e65 8 "38,0 38,4 50,4 30,2" "${vs[@]}"
else ./ab.sh e65-16 16 "70,6" "${vs[@]}"; ./ab.sh e65-32 32 "110,0 110,12" "${vs[@]}"
fi
