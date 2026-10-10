#!/usr/bin/env bash
# v52 per-entry timeline (trace only). e62.sh trace
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
v="td_v52:TD_ONEREL TD_ATOM_AR TD_UNIT_TRACE TD_CTA_TRACE"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$v"
for mc in "8 38,4" "8 38,0" "32 110,12"; do set -- $mc
  o=logs/u62-$1-${2/,/_}.pt
  CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 1 --shared 1 --numa-node 1 --trace $o >/dev/null 2>&1
  echo "######## $v M=$1 $2"; $PY ent_tl.py $o 0; $PY ent_tl.py $o 1 | tail -n +2
done
