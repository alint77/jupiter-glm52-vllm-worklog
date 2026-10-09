#!/usr/bin/env bash
# Run a command on a specific kdev hold:  kdj.sh <jobid> <cmd...>
cd /e/project1/profound/alint77/vllm
j=$1; shift
HOLD_JOB=$j exec agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "$@"
