#!/usr/bin/env bash
# v50 finisher sub-stamps (trace only). e60.sh trace
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
v="td_v50:TD_UNIT_TRACE TD_CTA_TRACE"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$v"
for mc in "8 38,4" "8 38,0" "32 110,12"; do set -- $mc
  o=logs/u60-$1-${2/,/_}.pt
  CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace $o >/dev/null 2>&1
  echo "######## $v M=$1 $2"; $PY trace_bd.py $o 4 8 | sed -n 2p; $PY fin_tl.py $o
done
