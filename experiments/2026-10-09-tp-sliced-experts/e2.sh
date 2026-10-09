#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
vs=(); for c in 12 16 20 24 32; do vs+=("td_v17:TD_COLD_CTAS=$c|--shared 1"); done
./ab.sh e2-32 32 "96,8 110,12 124,16" "${vs[@]}"
./ab.sh e2-8 8 "38,4 46,6" "${vs[@]}"
