#!/usr/bin/env bash
# v43 producer setup split, M=8 38/4, +/- record prefetch
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
a="td_v43:TD_NO_RECPF TD_UNIT_TRACE TD_CTA_TRACE"; b="td_v43:TD_UNIT_TRACE TD_CTA_TRACE"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$a" "$b" 2>&1 | grep -i error
for v in "$a" "$b"; do
  CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 1 --shared 1 --numa-node 1 --trace logs/u54.pt >/dev/null 2>&1
  echo "######## $v"; $PY trace_bd.py logs/u54.pt 4 8 | sed -n 2,3p; $PY prod_tl.py logs/u54.pt 2>/dev/null | tail -9; $PY setup_tl.py logs/u54.pt
done
