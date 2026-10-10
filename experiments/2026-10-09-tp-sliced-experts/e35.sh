#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V=("td_v32:TD_CTA_TRACE" "td_v32:TD_CTA_TRACE TD_COMPUTE_ONLY")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" 2>&1 | grep -i error
for v in "${V[@]}"; do for mc in "8 38,4" "32 110,12"; do set -- $mc
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m $1 --cell $2 --n 4 --shared 1 --numa-node 0 --trace logs/u35.pt >/dev/null 2>&1
  echo "## $v M=$1 $2"; $PY warp_tl.py logs/u35.pt
done; done
