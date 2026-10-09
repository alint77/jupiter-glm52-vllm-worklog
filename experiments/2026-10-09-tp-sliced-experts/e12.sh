#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
vs=(); for g in 1 2 3 4; do vs+=("td_v25:TD_GR1=$g|--shared 1"); done
./ab.sh e12-32 32 "110,12 96,8 110,0" "${vs[@]}"
./ab.sh e12 8 "38,4 46,6 38,0" "${vs[@]}"
