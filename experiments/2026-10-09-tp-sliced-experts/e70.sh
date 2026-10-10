#!/usr/bin/env bash
# shipped td_v55 (route_prep wrap/bound fixes) vs td_v54: kernel timing. e70.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
vs=("td_v54" "td_v55")
vs=("${vs[@]/%/|--shared 1}")
$PY -c "import kdev; kdev.build_many(['td_v54', 'td_v55'])"
if [[ $1 == 8 ]]; then ./ab.sh e70-8 8 "38,0 38,4 50,4 30,2" "${vs[@]}"
else ./ab.sh e70-16 16 "70,6" "${vs[@]}"; ./ab.sh e70-32 32 "110,0 110,12" "${vs[@]}"
fi
