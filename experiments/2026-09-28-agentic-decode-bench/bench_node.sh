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
# Logs and rows go to fscratch: the project1 inode quota is shared and full.
OUT="${BENCH_OUT:-/e/fscratch/profound/${USER}/agentic-bench}"; mkdir -p "${OUT}"
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
"${cmd[@]}" >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null || true; kill ${pid} 2>/dev/null || true; wait ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}/server-${tag}".{out,err} | sort -u | head; exit 1; }
  sleep 5
done
name=$(curl -fsS http://127.0.0.1:8027/v1/models | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "server ready $(date +%T) model ${name}"
# GREEDY_CHECK=1: temperature-0 completions of fixed ~600/~900-token prompts,
# saved for comparing outputs across arms.
if [[ -n "${GREEDY_CHECK:-}" ]]; then
  .venv/bin/python - "${name}" "${OUT}/greedy-${tag}.json" <<'PY'
import json, sys, urllib.request
text = open("README.md").read()
out = {}
for chars in (2400, 3600):
    body = {"model": sys.argv[1], "prompt": text[:chars], "max_tokens": 48,
            "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8027/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    out[chars] = {"text": r["choices"][0]["text"],
                  "prompt_tokens": r["usage"]["prompt_tokens"]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("greedy", {k: v["prompt_tokens"] for k, v in out.items()})
PY
fi
# PREFILL_SWEEP="512 1024 ...": TTFT on random prompts (1 output token) at each
# new-token count, before the agentic requests.
for L in ${PREFILL_SWEEP:-}; do
  .venv/bin/vllm bench serve --base-url http://127.0.0.1:8027 --model "${name}" \
    --tokenizer "/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16" \
    --dataset-name random --random-input-len "${L}" --random-output-len 1 \
    --num-prompts "${PREFILL_PROMPTS:-10}" --num-warmups 2 --max-concurrency 1 \
    --seed "${L}" --percentile-metrics ttft --save-result --result-dir "${OUT}" \
    --result-filename "prefill-${tag}-${L}.json" 2>&1 | grep -E "Mean TTFT|Median TTFT"
done
[[ -n "${SKIP_AGENTIC:-}" ]] || .venv/bin/python "${B}/agentic_bench.py" --model "${name}" --out "${OUT}/rows-${tag}.jsonl" --tasks agent_space/experiments/2026-09-26-mimo-routing-profile/tasks-{0,1,2,3}.json "${prof[@]}" "$@"
echo "=== done $(date +%T)"
