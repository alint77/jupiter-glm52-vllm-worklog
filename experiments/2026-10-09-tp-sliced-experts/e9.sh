#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
M="TD_ABL_NOMMA TD_ABL_NOFLUSH"
vs=("td_v24:$M|--shared 1" "td_v24:$M TD_STAGES=3|--shared 1" "td_v24:$M TD_GR1=4|--shared 1" "td_v24:$M TD_COLD_CTAS=32|--shared 1" "td_v24:$M TD_NO_PREFETCH_CLAIM|--shared 1" "td_v24:$M|--shared 0")
./ab.sh e9 8 "38,0 38,4" "${vs[@]}"
./ab.sh e9-32 32 "110,0 110,12" "${vs[@]}"
