# Claude Code AutoRound host

`server.sbatch` serves GLM-5.2 AutoRound W4G64 with MTP3 on one JUPITER
Booster node (TP4/EP4, c1). It uses the validated 5 GB HBM reserve, FP8 MLA KV
cache, full decode graphs, NUMA-local Grace offload, prefix caching, and the
native Anthropic Messages API.

From the vLLM checkout:

```bash
./claude-local.sh
```

The wrapper submits or reuses the four-hour job, waits for the authenticated
endpoint, and launches Claude Code with per-request reasoning effort set to
`max`. Pass `--effort LEVEL` to override it for a session. Lifecycle commands:

```bash
./claude-local.sh --start
./claude-local.sh --status
./claude-local.sh --monitor
./claude-local.sh --abort
./claude-local.sh --stop
```

`--abort` cancels a runaway generation without stopping or reloading the
model. Auto tool calls use GLM's parser without strict structural-tag
constraints, avoiding the xgrammar failure seen with long Claude turns.
