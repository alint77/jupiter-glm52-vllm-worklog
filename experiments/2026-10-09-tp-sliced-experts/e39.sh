#!/usr/bin/env bash
# ncu on v32 at M=8 38/4: e39.sh <full|loads>
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
if [[ $1 == full ]]; then ./ncu1.sh v32-full "td_v32" 8 38,4 --shared 1
else ./ncu1.sh v32-loads "td_v32:TD_ABL_NOMMA TD_ABL_NOFLUSH" 8 38,4 --shared 1; fi
