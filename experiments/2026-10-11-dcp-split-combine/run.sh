#!/usr/bin/env bash
# On a hold, from the dcp-fused worktree:  run.sh <log> <cmd...>
W=/e/fscratch/profound/${USER}/worktrees/dcp-fused
cd $W
export PYTHONPATH=$W VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev-fused TMPDIR=/tmp/rt$$
mkdir -p $TMPDIR
log=$1; shift
"$@" > $log 2>&1
