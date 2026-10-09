#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
./ncu1.sh v17co td_v17:TD_COMPUTE_ONLY 8 38,4 --shared 1 2>&1 | head -60
./tr.sh 8 38,4 td_v17
