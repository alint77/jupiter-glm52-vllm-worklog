#!/usr/bin/env bash
# What slows delivery in the full kernel: smem reads vs math (M=8 38/4)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
T="TD_UNIT_TRACE TD_CTA_TRACE"
V=("td_v35:$T TD_ABL_LDSONLY|ldsonly" "td_v35:$T TD_ABL_NOLDS|nolds" "td_v35:$T TD_ABL_NODECODE|nodecode" "td_v35:$T|full")
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]%%|*}" 2>&1 | grep -i error
for e in "${V[@]}"; do IFS='|' read v tag <<< "$e"
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 0 --trace logs/u42-$tag.pt >/dev/null 2>&1
  echo "######## $v"; $PY trace_bd.py logs/u42-$tag.pt 4 8 | grep -E "kernel end|consume R|wait latency R|late lat|tail"; $PY unit_tl.py logs/u42-$tag.pt 4 | grep -E "^R0|^R1|^ +consume" ; $PY unit_tl.py logs/u42-$tag.pt 4 | awk 'NR>12 && $1>=20 && $1<48 {h+=$2; n++} END {print "hot GB/s, 20-48 us mean:", h/n}'
done
