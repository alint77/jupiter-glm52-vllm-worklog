# Prefill + decode at c=4, under real concurrent load

**Date:** 2026-09-02  **Jobs:** 1626566/1626567 (died), 1626604/1626605

Every acceptance arm in this worklog issues requests **sequentially**, so no
tok/s figure recorded before today is an aggregate-under-load number — the
5.7046 AL / 136.2 tok/s and friends are single-stream results from a server
that merely *permitted* concurrency 4. This directory measures real c=4 load
with `vllm bench serve --max-concurrency 4 --request-rate inf` on the
16K-in / 1024-out PyTorch coding suite.

`arm-perf.sh` is adapted from
`2026-08-29-glm53-routing-capture/arm-realcode-short.sh` with two changes:
DCP is an argument rather than `[[ concurrency -gt 1 ]] && dcp=4` (that reflex
forces the shape DFlash2 is broken on), and the target is W4A16 with the
`glm53-w4a16-2496` profile to match `claude-glm53-c4-df2.sh`.

## Result: MTP3 / DCP4 / c=4 / 350K context

| metric | value |
| --- | ---: |
| output throughput | **209.3 tok/s** (total 316.4, 0.204 req/s) |
| TTFT | mean 552 ms, p50 **477**, p99 1073 |
| TPOT | mean **18.26 ms**, p50 18.55 |
| ITL | mean 53.96 ms (per *chunk*: MTP3 verifies 4 tokens/step) |
| KV | 1,400,255 tokens, 4.00x concurrency |
| residency | 2641 hot / 2159 cold experts per rank |

**This reproduces the historical reference**, which is the point of running it:
`2026-08-29-glm53-c4/server.sbatch` records 213.1 tok/s aggregate, 18.77 ms
TPOT and 477 ms TTFT from an earlier campaign, against 209.3 / 18.26 / 477
here. The harness is trustworthy.

## DFlash2 / DCP1 / c=4 did not run, and the reason is structural

Three launches died with `ValueError: Fixed allocations and reserve exceed HBM
capacity` (`tiered_moe_planner.py:328`). The cause is **not** the
`--kv-cache-memory` value, which is what the first two fixes wrongly chased.

`tiered_moe_physical.py:227` puts the MLA cache into the planner's *fixed*
allocations:

```python
fixed_hbm_allocations["main_mla_cache"] = kv_plan.main_cache_bytes
```

and `kv_plan` is sized from `max_model_len`, `max_num_seqs` and
`dcp_world_size` — not from the flag. So at DCP 1 the planner reserves
`max_num_seqs x max_model_len` of *replicated* MLA cache per rank:

| run | planner reserves per rank | outcome |
| --- | --- | --- |
| `repl2-eager-dcp1` (c=1, 400K) | 400K tokens | worked |
| `g-dcp1-c4` (c=4, 32K) | 128K tokens | worked |
| `df2-dcp1-c4` (c=4, 350K) | **1.4M tokens, ~78 GiB** | fails against 95 GiB |

The tiered-MoE validator's `max_num_seqs > 1 requires DCP (the replicated 400K
MLA cache does not fit more than one sequence per rank)` is stating exactly
this. `VLLM_TIERED_MOE_RELAX_SHAPE` lifts the assertion, not the memory: it
converts a clean config error into a planner overrun.

**Consequence: at DCP1, context and concurrency are not independent.** A 450K
total budget at c=4 means `max_model_len ~= 112K`, not 350K. The only proven
DCP1 capacity is 400K tokens, so c=4 there caps context near 100K.

## The trade this leaves

| | context x concurrency | throughput | acceptance |
| --- | --- | ---: | ---: |
| MTP3 @ DCP4 | 350K x 4 | **209.3 tok/s** | ~4.9 AL |
| DFlash2 @ DCP1 | ~100K x 4 | unmeasured | 5.70 AL |
| DFlash2 @ DCP4 | 350K x 4 | unmeasured | 3.51 AL (broken) |

DFlash2's acceptance advantage is only available at DCP1, and DCP1 c=4 costs
3.5x the context per request. Until the DCP4 acceptance deficit is fixed, MTP3
is the production choice on this shape — and fixing that deficit is what would
let DFlash2 keep both the acceptance and the context.

## Reproduce

```bash
bash submit.sh                       # both arms
# or one:
bash arm-perf.sh <label> dflash2|mtp3 <concurrency> <dcp>
```
