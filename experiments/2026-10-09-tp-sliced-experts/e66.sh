#!/usr/bin/env bash
# EP (prod whole-expert kernel) vs TP-sliced v53 variants over round 1's cell grids, one node
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
EP8="4,0 4,1 4,2 4,3 6,0 6,1 8,0 8,1 8,2 8,3 10,0 10,1 12,0 12,1 12,2 12,3 14,0 14,1 16,0 16,1 16,2 16,3 20,0 20,1 20,2"
EP32="16,0 16,2 16,4 16,6 20,0 20,2 20,8 24,0 24,2 24,4 24,6 28,0 28,2 28,8 32,0 32,2 32,4 32,6 40,0 40,2 40,4 40,8"
SL8="24,0 24,2 24,4 24,6 24,8 30,0 30,2 30,4 36,0 36,2 36,4 36,6 36,8 40,0 40,2 40,4 44,0 44,2 44,4 44,6 44,8 50,0 50,2 50,4 50,6 56,0 56,2 56,8"
SL32="72,4 72,8 72,12 72,20 84,4 84,8 84,16 84,24 96,4 96,8 96,12 96,20 104,4 104,8 104,16 112,4 112,8 112,12 112,20 128,4 128,8 128,12 128,16 128,24 144,4 144,8"
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
O=logs/ep-vs-slice-v53; mkdir -p $O
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" td_v53 "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT" "td_v53:TD_NOAHEAD"
( run 0 $PY bench_slice.py --time --variant full-1024x4 --m 8 --numa-node 0 --cells $EP8 > $O/ep8.jsonl 2>$O/ep8.err
  run 0 $PY bench_slice.py --time --variant full-1024x4 --m 32 --numa-node 0 --cells $EP32 > $O/ep32.jsonl 2>$O/ep32.err ) &
g=1
for v in td_v53 "td_v53:TD_NOAHEAD TD_NOAHEAD_HOT" "td_v53:TD_NOAHEAD"; do
  ( for m in 8 32; do c=SL$m
      run $g $PY kdev.py bench --v "$v" --m $m --numa-node $g --shared 1 --cells ${!c} 2>>$O/sl$m-$g.err | grep "^{" > $O/sl$m-$g.jsonl
    done ) &
  g=$((g + 1))
done
wait
cat $O/ep8.jsonl $O/ep32.jsonl $O/sl*-*.jsonl > $O/all.jsonl
wc -l $O/*.jsonl
