#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-decprof
#SBATCH --output=agent_space/experiments/2026-09-26-mimo-decode-profile/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-26-mimo-decode-profile/slurm-%j.err

# Torch-profiler traces of MiMo-V2.6 decode in the production configuration
# (serve.sbatch: DFlash k=7, R3, LM-only, prefill prefetch) plus the routing
# profile, at short and ~96K context. Prefix caching stays off (serve.sbatch
# default) so the 96K request really prefills.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-26-mimo-decode-profile
trace_root="/e/project1/profound/alint77/traces/mimo26-decode-${SLURM_JOB_ID}"
mkdir -p "${trace_root}"
echo "node $(hostname) traces ${trace_root}"

# record_shapes gives operand shapes for tier attribution; no Python stacks,
# which make large traces unloadable; ignore_frontend keeps API-server work out.
profiler="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_root}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"
export MIMO_PORT=8027
# Placement is serve.sbatch's default (routing profile + 1500 replicas) unless
# PLACEMENT_ARGS says otherwise; job 2074303 used
#   --tiered-moe-placement-profile .../profile-3827.json (no replicas, SP-MoE).
export MIMO_EXTRA_ARGS="--language-model-only ${PLACEMENT_ARGS:-} --profiler-config ${profiler}"

sed -e 's/^exec \.venv/.venv/' -e '/^#SBATCH/d' \
  agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch > "${trace_root}/serve-inline.sh"
bash "${trace_root}/serve-inline.sh" >"${trace_root}/server.out" 2>"${trace_root}/server.err" &
pid=$!
trap 'kill ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${trace_root}"/server.{out,err} | sort -u | head; exit 1; }
  sleep 5
done
echo "server ready $(date +%T)"
tr '\r' '\n' < "${trace_root}/server.out" | grep -ohE "Tiered MoE residency:.{0,120}" | sort -u

.venv/bin/python "${E}/capture.py" \
  --long-prompts agent_space/experiments/2026-09-04-mtp3-profile/prompts.jsonl \
  --short-prompts agent_space/experiments/2026-08-29-glm53-routing-capture/prompts-short.jsonl \
  --trace-root "${trace_root}" --temperature "${TEMPERATURE:-1.0}" ${CAPTURE_ARGS:-}
cp "${trace_root}/capture-report.json" "${E}/capture-report-${SLURM_JOB_ID}.json"
du -sh "${trace_root}"/decode-* | tee "${E}/trace-sizes-${SLURM_JOB_ID}.txt"
echo "=== done ==="
