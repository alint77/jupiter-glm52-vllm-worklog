#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
L="TD_ABL_NOMMA TD_ABL_NOFLUSH"
vs=("td_v26|--shared 1" "td_v26:TD_ABL_NOREADY|--shared 1" "td_v26:$L|--shared 1" "td_v26:$L TD_ABL_NOREADY|--shared 1")
./ab.sh e15-32 32 "110,12 110,0" "${vs[@]}"
./ab.sh e15 8 "38,4 38,0" "${vs[@]}"
