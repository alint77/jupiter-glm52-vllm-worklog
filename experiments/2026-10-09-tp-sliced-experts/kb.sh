#!/usr/bin/env bash
# Build variants on a hold and print their ptxas reports:  kb.sh <variant>...
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
P=/e/project1/profound/alint77/vllm/.venv/bin/python
for v in "$@"; do ( $P kdev.py build --v "$v" 2>&1 | grep -E "error|registers|spill|built" | grep -v "0 bytes spill" ) & done; wait
