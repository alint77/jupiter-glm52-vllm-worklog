#!/usr/bin/env bash
# One model, production serving config, the MiMo capture task set (agentic_bench.py).
#   ./onnode.sh <E>/bench_node.sh glm|mimo <tag> [agentic_bench.py args...]
# GLM: ../2026-09-27-glm53-mtp7-profile/serve.sh (DCP4, MTP7, tiered MoE) plus
#   what production (../2026-09-04-glm53-c1-df2/server.sbatch) adds for an agent
#   client: prefix caching, glm47 tool parser, glm45 reasoning parser.
# MiMo: ../2026-09-22-mimo-v26-pro/serve.sbatch with claude-server.sbatch's
#   MIMO_EXTRA_ARGS (prefix caching, mimo parsers, no 2048 output cap).
# Extra env for arms passes through (e.g. VLLM_* switches).
set -euo pipefail
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
model="$1"; tag="$2"; shift 2
case "${model}" in
  glm)
    export PREFIX_CACHING="${PREFIX_CACHING-1}"  # PREFIX_CACHING= : off
    export SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 ${SERVE_EXTRA:-}"
    cmd=(bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh)
    name=glm53-w4a16-tiered ;;
  mimo)
    export MIMO_PORT=8027
    export MIMO_EXTRA_ARGS="--language-model-only --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo --override-generation-config {\"max_new_tokens\":null}"
    cmd=(bash -c "sed -e 's/^exec \\.venv/.venv/' agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch | bash")
    name=$(grep -o -- '--served-model-name [^ ]*' agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch | awk '{print $2}' | head -1) ;;
  *) echo "model must be glm or mimo" >&2; exit 2 ;;
esac
prof=()
if [[ -n "${TRACE_ROOT:-}" ]]; then  # serve.sh reads TRACE_ROOT itself
  mkdir -p "${TRACE_ROOT}"
  pc="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TRACE_ROOT}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"
  [[ "${model}" == mimo ]] && export MIMO_EXTRA_ARGS="${MIMO_EXTRA_ARGS} --profiler-config ${pc}"
  prof=(--profile "${PROFILE_WINDOWS:-4}" --trace-root "${TRACE_ROOT}")
fi
"${cmd[@]}" >"${B}/server-${tag}.out" 2>"${B}/server-${tag}.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null || true; kill ${pid} 2>/dev/null || true; wait ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${B}/server-${tag}".{out,err} | sort -u | head; exit 1; }
  sleep 5
done
name=$(curl -fsS http://127.0.0.1:8027/v1/models | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "server ready $(date +%T) model ${name}"
.venv/bin/python "${B}/agentic_bench.py" --model "${name}" --out "${B}/rows-${tag}.jsonl" --tasks agent_space/experiments/2026-09-26-mimo-routing-profile/tasks-{0,1,2,3}.json "${prof[@]}" "$@"
echo "=== done $(date +%T)"
