# Phase 47 — hot-expert ranking from real Claude Code usage (GLM-5.3 W4A16)

The shipped GLM-5.3 ranking (`agent_space/profiles/glm53-w4a16-2496.json`) was
derived in Phase 44 from a *synthetic* agentic-coding driver: 16 scripted tasks,
23 turns, 389 traces, 129,392 routed positions. It was worth +8.30% ± 1.81%
decode over the GLM-5.2 placeholder it replaced. This phase re-derives it from
the real thing.

## Why real usage should beat the driver

The driver's tasks were written to *look* like agentic coding. Real Claude Code
turns differ in ways that plausibly move the routing distribution:

- far longer contexts (100k+ vs a few thousand), so more of the generation is
  conditioned on retrieved file content;
- a real system prompt and real tool schemas in every prefill;
- long tool-result spans (file reads, grep output, test logs) that the driver
  only approximated;
- the actual mix of reasoning, code, prose and JSON tool arguments.

The GLM-5.2 live capture recorded 154 requests in about an hour of ordinary use,
with responses up to 2,230 tokens (3,356 routed positions). An afternoon of
normal work therefore yields more real data than the whole synthetic run.

## Capture host

`server.sbatch` is the prod c4 launcher plus `--enable-return-routed-experts`.
Two deviations from prod are forced by the implementation, not chosen:

| | prod (`2026-08-29-glm53-c4`) | capture |
| --- | --- | --- |
| model runner | V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) | V1 |
| decode context parallel | 4 | 1 |
| `--max-num-seqs` | 4 | 1 |

`vllm/v1/worker/gpu/model_runner.py` — the V2 runner — contains no
`routed_experts` support, so `gpu_worker.py`'s `init_routed_experts_capturer()`
would fail on it. `Scheduler.__init__` separately asserts `dcp_world_size == 1`
when `enable_return_routed_experts` is set.

Neither deviation changes which experts the router picks: routing is the gate's
top-k over hidden states, and DCP shards attention KV without altering the math.
The cost is concurrency, and it cascades: `config/vllm.py:2325` rejects
`max_num_seqs > 1` under DCP1 outright — *"the replicated 400K MLA cache does not
fit more than one sequence per rank"* — and `:2337` pins `max_model_len` to
exactly 400000, so the context cannot be shortened to buy headroom. **The capture
host serves one request at a time.** Claude Code's parallel calls (subagents,
title/summary requests) queue behind the foreground turn, so live capture is
appreciably laggier than prod. This is why Phase 44's capture also ran at c=1.

Everything else matches prod: GLM-5.3-W4A16 (int4 group 32), the 2496-slot
profile, MTP3, prefix caching, `--gpu-memory-utilization 0.94`, KV pinned at
21,689,598,771 bytes. c=1 with MTP3 verifies 4 tokens per step, so the graph
shapes are `[4]` rather than prod's `[4,8,12,16]`.

## What is recorded

One `.npy` per request, shape `[routed_positions, 78, 8]` (int expert IDs), plus
a `manifest.jsonl` line. `routed_experts_prompt_start` is set to
`len(prompt) - 1` in the serving layer, so a trace begins at the last prompt
token: generation positions only, and a conversation prefix is never counted
twice across turns. `VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4` trims the ragged
prefix rows left by MTP3's 4-token verification steps.

**No prompts, responses, or tool arguments are written — only expert IDs.**

Traces land on fscratch (`caches/routes/glm53-cc-<jobid>`), not GPFS, because
they are many small files.

## Use

```bash
./claude-glm53-cap.sh              # Claude Code against the capture host
./claude-glm53-cap.sh --routes     # requests / routed positions captured so far
./claude-glm53-cap.sh --rank OUT   # ranking + profile from what is captured
```

`--rank` runs Phase 44's `build_profile.sh`, which calls `traces_to_manifest.py`
(hash-stable train/holdout split) and then `optimize_routing_profile.py` in both
`frequency` and `tail` residency modes.

## Status

- 2026-08-30: job 1535646 failed config validation at 81 s — submitted with
  `--max-num-seqs 4`, which DCP1 forbids. Resubmitted at c=1 as job 1535650.
