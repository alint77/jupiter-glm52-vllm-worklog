# Claude Code on a c1 Booster API

## Result

GLM-5.2 W4A16 is available to Claude Code on the JUPITER login node through
vLLM's native Anthropic Messages API. The first four-hour service is Slurm job
`1010645` on `jpbo-005-13`, ending at `2026-07-22 06:45:28 CEST`.

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

Claude Code 2.1.217 used `/v1/messages?beta=true` successfully. A synthetic
forced `tool_choice=any` request repeated the valid tool call until its token
limit, while the normal automatic tool choice used by the tested Claude Code
flow emitted exactly one call and stopped with `tool_use`.
