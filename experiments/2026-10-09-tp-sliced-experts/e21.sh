#!/usr/bin/env bash
# v29 phase probe: 38/4 at M=8 and 110/0 at M=32, eager calls
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v29:TD_PROBE >/dev/null 2>&1
run 0 $PY kdev.py once --v td_v29:TD_PROBE --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/p29-8.pt >/dev/null 2>&1 &
run 1 $PY kdev.py once --v td_v29:TD_PROBE --m 32 --cell 110,0 --n 4 --shared 1 --numa-node 1 --trace logs/p29-32.pt >/dev/null 2>&1 &
wait
for f in logs/p29-8.pt logs/p29-32.pt; do echo "## $f"; $PY probe29.py $f; done
