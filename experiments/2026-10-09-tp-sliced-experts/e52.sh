#!/usr/bin/env bash
# v41 (no stack frame in the producer) +/- record prefetch vs v39. e52.sh <8|32|trace>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v39" "td_v41:TD_NO_RECPF" "td_v41")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then
  $PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v41 "td_v41:TD_NO_RECPF" "td_v41:TD_NO_RECPF TD_UNIT_TRACE TD_CTA_TRACE" "td_v41:TD_UNIT_TRACE TD_CTA_TRACE"
  for v in td_v41 "td_v41:TD_NO_RECPF"; do
    n=$($PY -c "import kdev,sys; print(kdev.ext_name(sys.argv[1]))" "$v")
    echo "$v: $(grep -o 'Used 1[0-9][0-9] registers[^\n]*' $VLLM_CACHE_ROOT/kdev/$n/build.log)"
    echo "check $v"; CUDA_VISIBLE_DEVICES=0 $PY kdev.py check --v "$v" --reps 30 2>&1 | tail -1
  done
  ./ab.sh e52 8 "38,0 38,4 50,4" "${vs[@]}"
elif [[ $1 == 32 ]]; then ./ab.sh e52-16 16 "70,6" "${vs[@]}"; ./ab.sh e52-32 32 "110,0 110,12" "${vs[@]}"
else
  for v in "td_v41:TD_NO_RECPF TD_UNIT_TRACE TD_CTA_TRACE" "td_v41:TD_UNIT_TRACE TD_CTA_TRACE"; do
  for mc in "8 38,4" "32 110,12"; do set -- $mc
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace logs/u52.pt >/dev/null 2>&1
    echo "######## $v M=$1 $2"; $PY trace_bd.py logs/u52.pt 4 8 | head -24; $PY prod_tl.py logs/u52.pt 2>/dev/null
  done; done
fi
