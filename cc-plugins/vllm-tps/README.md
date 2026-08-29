# vllm-tps

Live throughput readout for a self-hosted vLLM backend, for Claude Code sessions
launched by `claude-local.sh` and its per-model wrappers (`claude-glm53-c4.sh`).

Two pieces:

- **`statusline.sh`** — renders decode throughput in the status line.
- **`/tps`** — a one-shot snapshot with lifetime prefill/decode rates,
  speculative-decode acceptance, prefix-cache hit rate and KV usage.

## What it shows

While a response is streaming:

```
GLM-5.3 glm53-c4 · vllm · ▸ 214 tok/s · acc 71% · 2 req · kv 3%
```

Between requests, the last completed request's figures:

```
GLM-5.3 glm53-c4 · vllm · pf 3421 · dec 213.1 tok/s · acc 71%
```

Those are deliberately different measurements. `vllm:generation_tokens_total`
is bumped every engine step, so its delta over wall-clock is the aggregate
decode rate across everything in flight *right now* — but it collapses to zero
the moment you stop prompting. The idle figures come from the per-request
histograms (`request_{prefill,decode}_time_seconds_sum` against
`request_{prefill_kv_computed,generation}_tokens_sum`), which accumulate per
completed request, so token delta over time delta is work divided by the time
that work actually took: independent of scrape cadence and of idle gaps.

`acc` is speculative-decode acceptance over the same window
(`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`),
which under MTP is the multiplier the draft head is actually buying you.

## Install

```bash
claude plugin marketplace add /e/project1/profound/alint77/vllm/agent_space/cc-plugins
claude plugin install vllm-tps@alint77-local
```

That gets you `/tps`. The status line is **not** a plugin component — Claude
Code only accepts it from a settings file — so wire it up once in
`~/.claude/settings.json`:

```json
"statusLine": {
  "type": "command",
  "command": "/e/project1/profound/alint77/vllm/agent_space/cc-plugins/vllm-tps/statusline.sh",
  "refreshInterval": 2,
  "padding": 0
}
```

`refreshInterval` is what makes the live number tick while a response streams;
without it Claude Code only repaints on session events.

## Configuration

The endpoint is read from `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`, which
`claude-local.sh` exports before exec'ing Claude Code. In a normal session
against Anthropic's API those are unset and the status line renders just the
model and directory.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_TPS_BASE_URL` | `$ANTHROPIC_BASE_URL` | Override the scrape target. |
| `VLLM_TPS_AUTH_TOKEN` | `$ANTHROPIC_AUTH_TOKEN` | Override the bearer token. |
| `VLLM_TPS_TIMEOUT` | `1` (`5` for `/tps`) | curl connect/total timeout, seconds. |
| `VLLM_TPS_MIN_INTERVAL` | `1` | Minimum seconds between scrapes. |
| `VLLM_TPS_WINDOW` | `2` | `/tps` live-rate measurement window, seconds. |
| `VLLM_TPS_STATE_DIR` | `$XDG_RUNTIME_DIR/claude-vllm-tps` | Per-session counter state. |

Requires `curl`, `jq` and `awk`.
