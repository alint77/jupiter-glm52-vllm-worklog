#!/usr/bin/env bash
# Where the 31 us skeleton goes: per-warp waits / producer spin, CO ablations, M=8 38/4
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V=("td_v32:TD_CTA_TRACE TD_COMPUTE_ONLY TD_ABL_NOMMA TD_ABL_NOFLUSH" "td_v32:TD_CTA_TRACE TD_COMPUTE_ONLY TD_ABL_NOMMA TD_ABL_NOFLUSH TD_ABL_NOREADY" "td_v32:TD_CTA_TRACE TD_COMPUTE_ONLY" "td_v32:TD_CTA_TRACE")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
for v in "${V[@]}"; do
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/u37.pt 2>&1 | grep -i "us\b\|error" | tail -2
  echo "## $v"; $PY warp_tl.py logs/u37.pt
done
