#!/usr/bin/env bash
# Ring depth probe: x rows reserved per stage (TD_XROWS) frees room for a 5th
# stage. e36.sh <8|32>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
L="TD_ABL_NOMMA TD_ABL_NOFLUSH"
vs=("td_v32" "td_v35:TD_XROWS=4" "td_v35:TD_XROWS=4 TD_STAGES=5" "td_v35:TD_XROWS=2 TD_STAGES=5"
    "td_v32:$L" "td_v35:$L TD_XROWS=4" "td_v35:$L TD_XROWS=4 TD_STAGES=5")
vs=("${vs[@]/%/|--shared 1}")
if [[ $1 == 8 ]]; then ./ab.sh e36 8 "38,0 38,4 50,4" "${vs[@]}"
else ./ab.sh e36-32 32 "110,0 110,12" "${vs[@]}"; fi
