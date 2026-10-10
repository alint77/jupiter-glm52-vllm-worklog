#!/usr/bin/env bash
# offload overhead: served kernel at the MTP3 mean mix (h, c) vs all hot (h + c, 0) vs hot only (h, 0)
# (sl_mix.py means: n=1 22/2.5, n=4 69/9.4, n=8 108/17.4, n=16 147/29.8)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
V="td_v57:TD_MAX_TOKENS=64"
./ab.sh e78-4 4 "22,3 25,0 22,0" "$V|--shared 1"
./ab.sh e78-16 16 "69,9 78,0 69,0" "$V|--shared 1"
./ab.sh e78-32 32 "108,17 125,0 108,0" "$V|--shared 1"
./ab.sh e78-64 64 "147,30 177,0 147,0" "$V|--shared 1"
echo E78_DONE
