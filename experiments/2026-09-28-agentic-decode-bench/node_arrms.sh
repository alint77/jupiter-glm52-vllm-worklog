#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
j=$1; tag=$2
until [[ "$(squeue -h -j $j -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
bash agent_space/experiments/2026-09-28-agentic-decode-bench/seq_arms.sh $j "$tag:VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm+FUSE_AR_RMS=true"
