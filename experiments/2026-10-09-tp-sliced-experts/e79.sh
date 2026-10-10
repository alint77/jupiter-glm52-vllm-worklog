#!/usr/bin/env bash
# Q2: why the served mix doesn't beat all-hot at M=8/16/32. Unit traces of the
# shipped kernel (v53 TD_NOAHEAD, SASS-identical) at mix / all-hot / hot-only.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
v="td_v53:TD_NOAHEAD TD_UNIT_TRACE TD_CTA_TRACE"
vb="td_v53:TD_NOAHEAD"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$v" "$vb"
run() {  # gpu m cells...
  g=$1 m=$2; shift 2
  for c in "$@"; do
    o=logs/u79-$m-${c/,/_}.pt
    CUDA_VISIBLE_DEVICES=$g numactl --cpunodebind=$g --membind=$g $PY kdev.py once --v "$v" --m $m --cell $c --n 1 --shared 1 --numa-node $g --trace $o >/dev/null 2>&1
    { echo "######## M=$m $c"; $PY trace_bd.py $o 4 8 | sed -n 2p; $PY tier_tl.py $o 10; } > logs/e79-$m-${c/,/_}.out
  done
}
run 0 8 40,5 45,0 40,0 &
run 1 16 69,9 78,0 69,0 &
run 2 32 108,17 125,0 108,0 &
wait
cat logs/e79-*.out
