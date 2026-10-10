#!/usr/bin/env bash
# v39 producer + consumer breakdown, M=8 38/4 and M=32 110/12
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
v="td_v39:TD_UNIT_TRACE TD_CTA_TRACE"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$v" 2>&1 | grep -i error
for mc in "8 38,4" "32 110,12"; do set -- $mc
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 0 --trace logs/u50-$1.pt >/dev/null 2>&1
  echo "######## M=$1 $2"; $PY trace_bd.py logs/u50-$1.pt 4 8; $PY hand_tl.py logs/u50-$1.pt; $PY prod_tl.py logs/u50-$1.pt
done
