#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
vs=(); for g in 1 2 4 6; do vs+=("td_v17:TD_GR1=$g|--shared 1"); done
./ab.sh e3-8 8 "38,0 30,0" "${vs[@]}"
./ab.sh e3-32 32 "110,0 96,0" "${vs[@]}"
