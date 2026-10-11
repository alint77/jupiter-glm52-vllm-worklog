#!/usr/bin/env bash
# Phase 1 checks on a hold (via onnode.sh). Logs to logs/.
#   run_tests.sh tests   -> DCP sparse tests, regs worktree (GPU 0) and the
#                           stock-library worktree (GPU 1), in parallel
#   run_tests.sh hashes  -> regs_hashes.py for prod, stock and regs, one GPU each
set -uo pipefail
E=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-11-flashmla-phase1
W=/e/fscratch/profound/${USER}/worktrees
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
L=${E}/logs
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton-fmla-p1
case $1 in
tests)
  for pair in "0 fmla-p1" "1 fmla-p1-stock"; do
    set -- ${pair}
    (cd ${W}/$2 && CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=${W}/$2 ${PY} -m pytest -q \
      tests/v1/attention/test_flashmla_sparse_dcp.py -p no:cacheprovider \
      > ${L}/tests-$2.log 2>&1; echo "EXIT $?" >> ${L}/tests-$2.log) &
  done
  wait ;;
hashes)
  i=0
  for lib in prod:/e/project1/profound/alint77/vllm/vllm/_flashmla_C.abi3.so \
             stock:/e/fscratch/profound/${USER}/build/fmla-partial/_flashmla_C.abi3.so \
             regs:/e/fscratch/profound/${USER}/build/fmla-regs/_flashmla_C.abi3.so; do
    CUDA_VISIBLE_DEVICES=${i} ${PY} ${E}/regs_hashes.py --lib ${lib#*:} \
      --out ${L}/hash-${lib%%:*}.json > ${L}/hash-${lib%%:*}.log 2>&1 &
    i=$((i+1))
  done
  wait
  for b in stock regs; do
    ${PY} ${E}/regs_hashes.py --compare ${L}/hash-prod.json ${L}/hash-${b}.json \
      > ${L}/compare-prod-${b}.log 2>&1; echo "EXIT $?" >> ${L}/compare-prod-${b}.log
  done ;;
esac
echo "RUN DONE $1"
