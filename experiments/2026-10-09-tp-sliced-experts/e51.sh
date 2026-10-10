#!/usr/bin/env bash
# v40 (record prefetch) vs v39. e51.sh <8|32|trace>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v39" "td_v40")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  echo "check td_v40"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v td_v40 --reps 30 2>&1 | tail -1
  ./ab.sh e51 8 "38,0 38,4 50,4" "${vs[@]}"
elif [[ $1 == 32 ]]; then ./ab.sh e51-16 16 "70,6" "${vs[@]}"; ./ab.sh e51-32 32 "110,0 110,12" "${vs[@]}"
else
  v="td_v40:TD_UNIT_TRACE TD_CTA_TRACE"
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$v" 2>&1 | grep -i error
  for mc in "8 38,4" "32 110,12"; do set -- $mc
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace logs/u51-$1.pt >/dev/null 2>&1
    echo "######## M=$1 $2"; $PY trace_bd.py logs/u51-$1.pt 4 8 | head -24; $PY hand_tl.py logs/u51-$1.pt; $PY prod_tl.py logs/u51-$1.pt 2>/dev/null
  done
fi
