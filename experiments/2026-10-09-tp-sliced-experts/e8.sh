#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
vs=(); for base in "" "TD_COMPUTE_ONLY "; do for a in "" "TD_ABL_NOMMA" "TD_ABL_NOFLUSH" "TD_ABL_NOMMA TD_ABL_NOFLUSH"; do
  d="$base$a"; d="${d% }"; [[ -z $d ]] && vs+=("td_v24|--shared 1") || vs+=("td_v24:$d|--shared 1"); done; done
./ab.sh e8 8 "38,4" "${vs[@]}"
./ab.sh e8-32 32 "110,12" "${vs[@]}"
