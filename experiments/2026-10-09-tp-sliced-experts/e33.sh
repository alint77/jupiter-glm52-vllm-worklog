#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "td_v33:TD_CTA_TRACE" "td_v33:TD_CTA_TRACE TD_COMPUTE_ONLY" 2>&1 | grep -i error
for v in "td_v33:TD_CTA_TRACE" "td_v33:TD_CTA_TRACE TD_COMPUTE_ONLY"; do
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/u33.pt >/dev/null 2>&1
  echo "## $v"; $PY epi_tl.py logs/u33.pt; $PY warp_tl.py logs/u33.pt | tail -2; $PY tl2.py logs/u33.pt | sed -n 1,4p
done
