#!/usr/bin/env bash
# First measurements on a kdev hold, all four GPUs (NUMA-bound each):
#   GPU 0: roofline, then M=8: v0 + side-stream shared (today) vs v3
#   GPU 1: M=8: v1 (no shared), v2, v3 again (ordering check)
#   GPU 2: M=32: v0 + side shared vs v3;  GPU 3: M=32: v1, v2, v3
# then on GPU 0: v3 traced call (M=8 and M=32), ncu of v3 at M=8.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
mkdir -p $TMPDIR logs
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V0='td_v0:TD_INTER=512 TD_CHUNK0=1024 TD_STAGES0=4 TD_CHUNK1=512 TD_STAGES1=8'
C8="30,2 38,4 46,6 38,0"
C32="96,8 110,12 124,16 110,0"
run() {  # gpu args...
  local g=$1; shift
  CUDA_VISIBLE_DEVICES=$g numactl --cpunodebind=$g --membind=$g $PY kdev.py "$@" --numa-node $g 2>>logs/s1-gpu$g.err
}
# build everything first, serially (shared build dirs)
for v in "$V0" "td_v1:" "td_v2:" "td_v3:" "td_v4:" "td_v5:" "td_v5:TD_CTA_TRACE"; do
  CUDA_VISIBLE_DEVICES=0 $PY -c "import kdev; kdev.build('$v')" 2>>logs/s1-build.err >/dev/null
done
echo "built $(date +%T)"
{ run 0 roof; run 0 bench --v "$V0" --m 8 --cells $C8 --side-shared 1; run 0 bench --v td_v5: --m 8 --cells $C8 --shared 1; } > logs/s1-gpu0.jsonl &
{ run 1 bench --v td_v1: --m 8 --cells $C8; run 1 bench --v td_v2: --m 8 --cells $C8 --shared 1; run 1 bench --v td_v3: --m 8 --cells $C8 --shared 1; run 1 bench --v td_v4: --m 8 --cells $C8 --shared 1; run 1 bench --v td_v5: --m 8 --cells $C8 --shared 1; } > logs/s1-gpu1.jsonl &
{ run 2 bench --v "$V0" --m 32 --cells $C32 --side-shared 1; run 2 bench --v td_v5: --m 32 --cells $C32 --shared 1; } > logs/s1-gpu2.jsonl &
{ run 3 bench --v td_v1: --m 32 --cells $C32; run 3 bench --v td_v2: --m 32 --cells $C32 --shared 1; run 3 bench --v td_v3: --m 32 --cells $C32 --shared 1; run 3 bench --v td_v4: --m 32 --cells $C32 --shared 1; run 3 bench --v td_v5: --m 32 --cells $C32 --shared 1; } > logs/s1-gpu3.jsonl &
wait
echo "benched $(date +%T)"
for mc in "8 38,4" "32 110,12"; do
  set -- $mc
  run 0 once --v "td_v5:TD_CTA_TRACE" --m $1 --cell $2 --n 1 --shared 1 --trace logs/s1-trace-m$1.pt >/dev/null
  echo "== trace M=$1 cell $2"; $PY tl2.py logs/s1-trace-m$1.pt
done
CUDA_VISIBLE_DEVICES=0 ncu --target-processes all --profile-from-start off -k regex:"layer_kernel|route_prep|finalize" \
  --set full -o logs/s1-ncu-v5-m8 -f numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v td_v5: --m 8 --cell 38,4 --n 1 --shared 1 \
  >/dev/null 2>>logs/s1-ncu.err
ncu --import logs/s1-ncu-v5-m8.ncu-rep --page details 2>/dev/null > logs/s1-ncu-v5-m8.txt
grep -E "^  [a-z_]+|Duration|DRAM Throughput|Memory Throughput|Compute \(SM\)|L2 Cache Throughput|Issue Slots Busy|No Eligible|Registers Per|Achieved Occupancy|Stall" logs/s1-ncu-v5-m8.txt | head -60
echo "=== done $(date +%T)"
