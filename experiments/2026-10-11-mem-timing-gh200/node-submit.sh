#!/usr/bin/env bash
# Launch a script inside the mem-timing standby allocation.
#   node-submit.sh <gpu-index|all> <script> [args...]
# Blocks until the step finishes. GPU is pinned through CUDA_VISIBLE_DEVICES;
# the step claims gpu:4 with --overlap (the verified Booster pattern), so
# concurrent steps must pin different GPU indices. Log lands in logs/.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
cd "$repo_dir"
E=agent_space/experiments/2026-10-11-mem-timing-gh200
jobid="$(<"${E}/alloc.jobid")"
sel="${1:?usage: node-submit.sh <gpu-index|all> <script> [args]}"; shift
script="${1:?script path}"; shift
case "$sel" in
  all) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
  0|1|2|3) export CUDA_VISIBLE_DEVICES="${sel}" ;;
  *) echo "bad gpu selector: ${sel}" >&2; exit 2 ;;
esac
mkdir -p "${E}/logs"
name="$(basename "$script")"; name="${name%.*}"
log="${E}/logs/${name}-gpu${sel}-$(date +%H%M%S).log"
rc=0
srun --jobid="$jobid" --overlap --cpu-bind=none --unbuffered \
  --nodes=1 --ntasks=1 --gres=gpu:4 \
  bash "$script" "$@" >"$log" 2>&1 || rc=$?
echo "step_exit=${rc} log=${log}"
tail -n 40 "$log"
exit "$rc"
