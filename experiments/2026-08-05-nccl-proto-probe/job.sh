#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --job-name=nccl-proto-probe
#SBATCH --output=agent_space/experiments/2026-08-05-nccl-proto-probe/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-nccl-proto-probe/slurm-%x-%j.err
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
D=agent_space/experiments/2026-08-05-nccl-proto-probe
echo "node: $(hostname)"

for proto in unset LL LL128 Simple; do
  echo ""
  echo "######## NCCL_PROTO=${proto}"
  unset NCCL_PROTO
  [[ "${proto}" != "unset" ]] && export NCCL_PROTO="${proto}"
  # INIT+ENV subsystems print the protocol NCCL selected and any env override.
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV \
    .venv/bin/torchrun --standalone --nproc_per_node=4 "${D}/probe.py" \
    >"${D}/out-${proto}.txt" 2>"${D}/dbg-${proto}.txt" || echo "ARM ${proto} FAILED"
  grep -h '^{' "${D}/out-${proto}.txt" 2>/dev/null || true
  grep -ohiE "NCCL_PROTO set by environment[^ ]*.*|Using network [A-Za-z]+|NCCL version [0-9.]+" \
    "${D}/dbg-${proto}.txt" 2>/dev/null | sort -u | head -3 || true
done
echo ""
echo "=== done ==="
