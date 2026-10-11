#!/usr/bin/env bash
# Mutation check: the scheduled-decode tests must fail when the schedule drops
# the last partial block (floor instead of ceil). Runs in a throwaway copy of
# the stock-library worktree's sparse_utils.py, restored afterwards.
set -uo pipefail
E=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-11-flashmla-phase1
W=/e/fscratch/profound/${USER}/worktrees/fmla-p1-stock
F=${W}/vllm/v1/attention/backends/mla/sparse_utils.py
cp ${F} ${F}.orig
sed -i 's|num_blocks = tl.maximum((live + 63) // 64, 1)|num_blocks = tl.maximum(live // 64, 1)|' ${F}
grep -q "tl.maximum(live // 64, 1)" ${F} || { echo "mutation not applied"; mv ${F}.orig ${F}; exit 1; }
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton-fmla-p1-mut
cd ${W} && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=${W} /e/project1/profound/alint77/vllm/.venv/bin/python \
  -m pytest -q tests/v1/attention/test_flashmla_sparse_dcp.py -k "sched" -p no:cacheprovider \
  > ${E}/logs/tests-mutant.log 2>&1
echo "EXIT $?" >> ${E}/logs/tests-mutant.log
mv ${F}.orig ${F}
echo "MUTANT DONE"
