#!/usr/bin/env bash
# Re-trace with the unit full stamp taken before the trace atomic (v35)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
T="TD_UNIT_TRACE TD_CTA_TRACE"
V=("td_v35:$T|4|full" "td_v35:$T TD_XROWS=4 TD_STAGES=5|5|s5" "td_v35:$T TD_ABL_NOMMA TD_ABL_NOFLUSH|4|loads" "td_v35:$T TD_ABL_LDSONLY|4|ldsonly")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]%%|*}" 2>&1 | grep -i error
for e in "${V[@]}"; do IFS='|' read v s tag <<< "$e"
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/u43-$tag.pt >/dev/null 2>&1
  echo "######## $v"; $PY trace_bd.py logs/u43-$tag.pt $s 8; $PY wait_dist.py logs/u43-$tag.pt
done
