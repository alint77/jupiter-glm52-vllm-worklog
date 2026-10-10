#!/usr/bin/env bash
# same-node references for e80: all-hot and hot-only
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
V="td_v57:TD_MAX_TOKENS=64"
./ab.sh e81-16 16 "69,9 78,0 69,0" "$V|--shared 1"
./ab.sh e81-32 32 "108,17 125,0 108,0" "$V|--shared 1"
