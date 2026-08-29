# GLM-5.3 agentic-coding routing capture

Re-derives the tiered-MoE hot-expert ranking for GLM-5.3 from agentic coding
traffic. Every 5.3 placement profile shipped so far carries GLM-5.2's ranking:
`nvfp4-profile-2400.json`'s optimizer string ends in
`ported-to-nvfp4-PLACEHOLDER-RANKING-FROM-GLM-5.2`, and 5.3's post-training
moved the router, so which experts are hot has changed. Placement is
quality-neutral (0.59pp across 2800 relocated experts), so the stale ranking
costs throughput, not accuracy.

## Pipeline

1. **Capture host** — `server.sbatch`, launched via `../../../claude-glm53-capture.sh`.
   GLM-5.3 NVFP4 tiered, c1/TP4/EP4, MTP3, `--enable-return-routed-experts`.
   Writes one `.npy` of shape `(positions, 78, 8)` per request plus
   `manifest.jsonl` into `routes-<job>/`. Only expert IDs are stored — no
   prompts, responses, or tool arguments.

   Three constraints are load-bearing and were established by the 2026-07-26
   GLM-5.2 capture:
   - V1 model runner (`unset VLLM_USE_V2_MODEL_RUNNER`) — routed-expert return
     is not implemented on V2.
   - no `--decode-context-parallel-size` — DCP does not return routed experts.
   - `VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4` — c1 + MTP3 verifies 4 tokens per
     step, and the writer trims the ragged prefix so `positions % 4 == 0`, which
     `oracle.py` requires.

2. **Workload** — `run_agentic_capture.py` runs a real tool-calling agent loop
   over `agentic_tasks.json`: 16 coding tasks (23 user turns) that search, read,
   and edit against the actual vLLM tree, with writes confined to a sandbox.
   Traces start at generation, so what lands in the ranking is what the model
   *emits* under agentic coding: tool-call JSON, code, and reasoning about code.

3. **Manifest adapter** — `traces_to_manifest.py` converts the server's
   `manifest.jsonl` to the `manifest.json` the optimizer reads, dropping
   identity-fallback and short traces and assigning a hash-stable train/held-out
   split so re-running reproduces the same profile.

4. **Ranking** — `build_profile.sh` runs
   `agent_space/benchmarks/optimize_routing_profile.py` at 2400 hot slots per
   rank (matching the shipped profile's 9600 hot experts) in both `frequency`
   and `tail` residency modes, and stamps the NVFP4 `config_sha256` /
   `index_sha256` from the checkpoint manifest.

5. **Replicas** — `2026-07-31-replicated-expert-scheduling/oracle.py`
   `--budgets 985` promotes the version 1 profile to the version 2 layout with
   `secondary_ranks` (985 copies per rank = the 3940 non-`-1` entries the
   shipped profile carries). Its `EXPERT_BYTES` constant is the AutoRound
   W4G64 figure, 20,054,024; NVFP4 is 21,233,680. That affects only the
   reported `grace_gb_per_rank`, not the placement.

## Result

Capture job `1532971`: 389 traces, 377 usable, **129,392 routed positions**
across all 16 task families, no driver aborts. The shippable artefact is
`results-1532971/replicas-985.json` -- schema version 2, 9600 hot experts, 3940
replicas, NVFP4 fingerprints, accepted by the runtime validator. It is a
drop-in replacement for `nvfp4-profile-2400.json`.

Held-out split (89 requests, 20,500 positions):

| profile | cold-hit | cold-critical/token | tail |
| --- | --- | --- | --- |
| GLM-5.2 ranking (shipped) | 0.4047 | 138.4 | 179.40 |
| re-derived | 0.2290 | 87.6 | 115.75 |

For scale, a linear/even placement scores 0.4977, so the GLM-5.2 ranking was
recovering only about a tenth of the gap between no ranking and a correct one.
Hot-set overlap with the shipped profile is 0.629: 37% of the resident set sat
on the wrong experts.

Three things the evidence settles:

- **The ranking is the whole story, not the residency strategy.** Frequency,
  tail-aware, and layer-concentrated residency all converge to the same hot set
  (overlap 0.999-1.000). At `--mixed-layer-penalty 0.0` the layer-concentrated
  optimiser reproduces the frequency solution exactly.
- **It generalises past this task list.** A strict split by task family -- train
  on 9 families, evaluate on 163 requests from 7 unseen ones, no conversation
  crossing the split -- gives 0.4126 -> 0.2742 cold-hit and 140.2 -> 86.2
  cold-critical. That last figure matches the request-split profile (87.2) on a
  third less training data, and the two hot sets overlap 0.918.
- **Replicas still pay on top.** The oracle replay puts the routed span at
  11.174 -> 10.273 ms at c1 (-8.1%) and 28.317 -> 25.707 ms at c4 (-9.2%).

## Known limitation: capped generations

12 of 389 traces came back as the identity fallback -- experts 0..7 on every
routed layer, every position. The correlation is exact and one-directional:

| | identity | good |
| --- | --- | --- |
| ran to the `max_tokens` cap | 12 | 16 |
| finished naturally | 0 | 354 |

Every affected trace hit the cap, no trace that finished naturally was ever
affected, and only 43% of capped requests were hit. `traces_to_manifest.py`
drops them, so they cannot corrupt the ranking, but the corpus is mildly biased
against the longest generations. `build_claude_routing_grid.py` carries the
same `is_default_route_trace` guard, so the GLM-5.2 capture saw this too.

## Pipeline validation

The whole downstream chain was exercised on synthetic traces before the capture
ran, so only the capture itself was untested when it started:

| stage | result |
| --- | --- |
| `traces_to_manifest.py` | 12 traces, 10 train / 2 held out, domains joined |
| `optimize_routing_profile.py` | version 1 profile, 9600 hot experts |
| fingerprints | `config_sha256` and `index_sha256` match the shipped NVFP4 profile |
| `load_tiered_moe_placement_profile` | version 1 accepted by the runtime validator |
| `oracle.py --budgets 985` | version 2 profile, 3940 replicas, accepted by the validator |
