#!/usr/bin/env bash
# Warp 0 work split: full / compute-only / loads-only, M=8 38/4
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
T="TD_UNIT_TRACE TD_CTA_TRACE"
V=("td_v35:$T|full" "td_v35:$T TD_COMPUTE_ONLY|co" "td_v35:$T TD_ABL_NOMMA TD_ABL_NOFLUSH|loads")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]%%|*}" 2>&1 | grep -i error
for e in "${V[@]}"; do IFS='|' read v tag <<< "$e"
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 2 --shared 1 --numa-node 0 --trace logs/u44-$tag.pt >/dev/null 2>&1
  echo "######## $v"; $PY trace_bd.py logs/u44-$tag.pt 4 8 | sed -n 2,20p | grep -vE "^\s+(wait producer S|wait latency S)"; $PY hand_tl.py logs/u44-$tag.pt
done
