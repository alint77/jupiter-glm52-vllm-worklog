# Claude Code on a c1 Booster API

## Result

GLM-5.2 W4A16 is available to Claude Code on the JUPITER login node through
vLLM's native Anthropic Messages API. The live replacement service is Slurm
job `1010672` on `jpbo-005-13`, ending at `2026-07-22 07:25:04 CEST`.

From the vLLM checkout:

```bash
./claude-local.sh
```

The wrapper reuses the live job or submits a new four-hour Booster job, waits
for authenticated API readiness, exports the Claude variables, and launches
Claude Code. Useful lifecycle commands are:

```bash
./claude-local.sh --status
./claude-local.sh --start
eval "$(./claude-local.sh --env)"
./claude-local.sh --stop
```

The third command configures the current shell so a later bare `claude` command
uses the same local API.

For rolling throughput in a second login-node terminal:

```bash
./claude-local.sh --monitor
./claude-local.sh --monitor 1  # optional one-second interval
```

`PREFILL_TOK/S` counts newly computed KV tokens and excludes prefix-cache hits;
`DECODE_TOK/S` counts emitted generation tokens. Both are rolling wall-time
rates, so they read zero while the engine is idle. The monitor also shows
running and waiting request counts and stops if the server job exits.

Claude Code defaults unknown/custom model names to a 200,000-token context
window. The wrapper exports `CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000` so its UI
and auto-compaction budget match this server's `max-model-len=400000`.

## Deployment

- One Booster node with four GH200 GPUs, TP4/EP4, DCP1, and `max-num-seqs=1`.
- V1 runner, MTP3, full decode graphs, piecewise prefill graphs, FP8 MLA KV,
  prefix caching, NUMA-strict tiered MoE, and the c1/q4 `hybrid-p0.5` profile.
- Native `/v1/messages` and `/v1/messages/count_tokens`, with the `glm47`
  reasoning and tool-call parsers.
- The Booster service binds to its internal interface. The login node reaches
  the allocated node directly over JUPITER's common internal fabric; no SSH
  tunnel, reverse proxy, or compute-node internet access is involved.
- A random bearer token and job endpoint markers live under
  `/e/scratch/profound/naeimitabiei1/claude-local` with owner-only permissions.
  They are never committed. Runtime logs are ignored for the same reason.

The batch entry point is [`server.sbatch`](server.sbatch). It loads the JUPITER
modules through `agent_space/jupiter-env.sh`, uses only local model/cache paths,
starts vLLM, and publishes the endpoint only after its authenticated health
check passes. `claude-local.sh` performs a second health check from the login
node before returning or launching Claude.

## Validation

The first cold start took about 26 minutes, dominated by checkpoint loading,
followed by compilation and graph capture. Validation from the login node:

| Check | Result |
|---|---|
| Unauthenticated `/v1/models` | HTTP 401 |
| Authenticated `/v1/models` | HTTP 200 |
| Anthropic token counting | HTTP 200, 17 input tokens |
| Anthropic message | `JUPITER API READY` |
| Automatic Anthropic tool call | One `get_temperature({"city":"Berlin"})` call |
| Claude Code plain request | Success in one turn |
| Claude Code `Bash(pwd)` loop | Success in two turns; correct checkout path |
| Throughput monitor | 892 prefill tok/s; 99-122 steady decode tok/s |

Claude Code 2.1.217 used `/v1/messages?beta=true` successfully. A synthetic
forced `tool_choice=any` request repeated the valid tool call until its token
limit, while the normal automatic tool choice used by the tested Claude Code
flow emitted exactly one call and stopped with `tool_use`.

## Quantized chunked-context prefill fix

A repeated Claude request exposed an existing long/prefix-cached prefill bug in
the first job. The chunked-context MLA path correctly computed a fallback dtype
for quantized `kv_b_proj` layers without a `.weight` attribute, but the cast
still dereferenced `.weight`. That raised `AttributeError` on every TP rank and
ended job `1010645`.

Commit `fcd9fb65c` uses the already-computed fallback dtype. Job `1010672`
loaded the fix and passed two consecutive Claude Code prompts with a shared
prefix. The second request exercised chunked-context prefill, completed at
about 114 decode tok/s, and left the engine healthy. The full pre-commit suite
passed with its cache moved from quota-limited home storage to scratch.
