#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v27:TD_CTA_TRACE >/dev/null 2>&1
run 0 $PY kdev.py once --v td_v27:TD_CTA_TRACE --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/p27-8.pt >/dev/null 2>&1 &
run 1 $PY kdev.py once --v td_v27:TD_CTA_TRACE --m 32 --cell 110,0 --n 4 --shared 1 --numa-node 1 --trace logs/p27-32.pt >/dev/null 2>&1 &
wait
for f in logs/p27-8.pt logs/p27-32.pt; do echo "## $f"; $PY probe27.py $f; done
