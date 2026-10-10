#!/usr/bin/env bash
# unit timelines, M=8 38/4: full, loads only, compute only
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
T="TD_CTA_TRACE TD_UNIT_TRACE"
V=("td_v32:$T" "td_v32:$T TD_ABL_NOMMA TD_ABL_NOFLUSH TD_ABL_NOSHMMA")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
i=0
for v in "${V[@]}"; do i=$((i+1))
  for cell in 38,4 38,0; do
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell $cell --n 4 --shared 1 --numa-node 0 --trace logs/ut-$i-$cell.pt >/dev/null 2>&1
  echo "#### $v $cell"; $PY unit_tl.py logs/ut-$i-$cell.pt 2; $PY tl2.py logs/ut-$i-$cell.pt | tail -10
  done
done
