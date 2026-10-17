#!/usr/bin/env bash
# compute-sanitizer over our Hopper kernels' correctness suites, login GPU.
# Usage: sweep.sh [tools...]  (default: racecheck synccheck memcheck initcheck)
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh >/dev/null 2>&1
export PYTHONPATH=$PWD
E=agent_space/experiments/2026-10-17-kernel-verification
L=$E/logs/sweep; mkdir -p $L
T=tests/kernels/moe
declare -A CMD FILT
CMD[skinny]="$E/skinny_check.py";                           FILT[skinny]=skinny::
CMD[sliced]="-m pytest -q -x $T/test_tiered_decode_sliced.py -k 1-False||8-True||47-True||64-True"
FILT[sliced]=tiered_decode::
CMD[decode]="-m pytest -q -x $T/test_tiered_decode_moe.py -k 8-int4||32-mxfp4||padding||replica"
FILT[decode]=tiered_decode::
CMD[prefill]="-m pytest -q -x $T/test_tiered_prefill_moe.py -k sliced_tiers_match_fp32||two_tiers_match_fp32||heavy||padding||grace||tiny||schedules"
FILT[prefill]=tiered_prefill::
for tool in ${@:-racecheck synccheck memcheck initcheck}; do
  for s in skinny sliced decode prefill; do
    extra="--kernel-name kns=${FILT[$s]}"
    [[ $tool == memcheck ]] && extra="$extra --leak-check no"
    # a kernel filter hides other kernels' writes from initcheck, so every
    # torch-produced buffer would read as uninitialized
    [[ $tool == initcheck ]] && extra=""
    read -r -a argv <<< "${CMD[$s]}"
    for i in "${!argv[@]}"; do argv[$i]=${argv[$i]//||/ or }; done
    start=$(date +%s)
    compute-sanitizer --tool $tool $extra --print-limit 20 \
      .venv/bin/python "${argv[@]}" > $L/$s.$tool.log 2>&1
    rc=$?
    echo "$(date +%T) $s $tool rc=$rc $(( $(date +%s) - start ))s $(grep -h -E 'SUMMARY' $L/$s.$tool.log | tail -1) $(grep -h -E '[0-9]+ (passed|failed)' $L/$s.$tool.log | tail -1)" >> $L/summary.txt
  done
done
echo DONE >> $L/summary.txt
