#!/usr/bin/env bash
# td_v56 timing: M=32 regression vs td_v55 (32 and 64 builds), M=64 cells (sl_grid.py p5..p95)
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
V64="td_v56:TD_MAX_TOKENS=64"
./ab.sh e72-32 32 "96,8 108,16 120,32" "td_v55|--shared 1" "td_v56|--shared 1" "$V64|--shared 1"
./ab.sh e72-64 64 "132,16 144,24 144,40 156,48 156,60" "$V64|--shared 1"
echo E72_DONE
