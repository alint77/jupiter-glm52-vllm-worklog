#!/usr/bin/env bash
# w2-phase delivery: loads-only and full, GR1 1/2/4, NOREADY; benches + timelines (38/4)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
L="TD_ABL_NOMMA TD_ABL_NOFLUSH TD_ABL_NOSHMMA"
V=(td_v32 "td_v32:TD_GR1=2" "td_v32:TD_GR1=4" "td_v32:TD_ABL_NOREADY"
   "td_v32:$L" "td_v32:$L TD_GR1=2" "td_v32:$L TD_GR1=4" "td_v32:$L TD_ABL_NOREADY")
T=(); for v in "${V[@]}"; do [[ $v == *:* ]] && T+=("$v TD_CTA_TRACE TD_UNIT_TRACE") || T+=("$v:TD_CTA_TRACE TD_UNIT_TRACE"); done
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${V[@]}" "${T[@]}" 2>&1 | grep -i error
run() { CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 "${@:2}"; }
b() { g=$1 m=$2; shift 2; for v in "${V[@]}"; do for rep in 1 2; do run $g $PY kdev.py bench --v "$v" --m $m --numa-node $g --shared 1 --cells "$@" 2>/dev/null | grep "^{"; done; done; }
b 0 8 38,0 38,4 50,4 > logs/e29-g0.txt & b 1 32 110,0 110,12 > logs/e29-g1.txt &
( i=0; for v in "${T[@]}"; do i=$((i+1)); run 2 $PY kdev.py once --v "$v" --m 8 --cell 38,4 --n 4 --shared 1 --numa-node 2 --trace logs/u29-$i.pt >/dev/null 2>&1; echo "#### $v"; $PY unit_tl.py logs/u29-$i.pt 4 | sed -n 1,6p; $PY unit_tl.py logs/u29-$i.pt 4 | awk 'NR>7' | tr '\n' '|' | head -c 2000; echo; done ) > logs/e29-tl.txt 2>&1 &
wait
$PY abl_sum.py logs/e29-g0.txt logs/e29-g1.txt
cat logs/e29-tl.txt
