#!/usr/bin/env bash
# Unit + CTA timelines, v32 M=8 38/4: loads-only vs full
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V=("td_v32:TD_UNIT_TRACE TD_CTA_TRACE TD_ABL_NOMMA TD_ABL_NOFLUSH" "td_v32:TD_UNIT_TRACE TD_CTA_TRACE")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
for v in "${V[@]}"; do
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/u40.pt >/dev/null 2>&1
  echo "######## $v"; $PY tl2.py logs/u40.pt; $PY unit_tl.py logs/u40.pt 4
done
