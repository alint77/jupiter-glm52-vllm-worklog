# DFlash2 vs MTP at K=3 and K=7, on Claude-Code-shaped traffic

Four full-node Booster jobs, 2026-09-04, jobs 1664143-1664146. Identical in
everything but the speculator: GLM-5.3 W4A16 tiered, c=1, DCP1, 400K context,
utilisation 0.90, reserve 10, capture [K+1]. Sixty requests replayed verbatim
on each arm -- ten code tasks and ten prose tasks over three context
assemblies each, ~96K tokens per prompt, assembled from real fork sources and
worklog prose.

## Why this exists

The DFlash2 card reports 5.94 acceptance on GSM8K against MTP's 5.12, and
that ordering is what motivated the DFlash2 backport. But the live server's
counters over a real Claude Code session gave **2.53**, not 5.63. Short
arithmetic answers are far more predictable than agentic coding traffic, so
GSM8K cannot answer "which speculator should serve Claude Code".

## Results

| arm | AL | step ms | tok/s decode | TTFT med | e2e/req | hot experts |
|---|---|---|---|---|---|---|
| DFlash2 K=3 | 2.584 | 29.4 | 87.3 | 23.16s | 45.5s | 2224 |
| DFlash2 K=7 | 3.099 | 37.1 | 82.4 | 23.24s | 47.3s | 2224 |
| **MTP K=3** | **2.691** | **29.1** | **91.2** | 23.16s | **44.9s** | 2325 |
| MTP K=7 | 3.242 | 39.0 | 82.1 | 23.50s | 47.9s | 2325 |

### 1. MTP beats DFlash2 on acceptance at both widths

2.691 vs 2.584 at K=3, 3.242 vs 3.099 at K=7, and MTP is ahead at every
draft position. This **inverts the GSM8K ordering** that motivated the
DFlash2 work. On this traffic the lattice drafter does not win.

### 2. K=3 beats K=7 on throughput for both methods

K=7 buys ~20% more acceptance and spends it on a wider verify step:
29.4 -> 37.1 ms for DFlash2, 29.1 -> 39.0 ms for MTP. Net decode throughput
falls 5.6% (DFlash2) and 10.0% (MTP). At K=7 the two methods are a **tie**
on throughput (82.4 vs 82.1) despite MTP's higher acceptance, because MTP's
step is slower. MTP's advantage is real only at K=3.

### 3. The methods respond to content type in opposite directions

| arm | code AL | prose AL | code tok/s | prose tok/s |
|---|---|---|---|---|
| DFlash2 K=3 | 2.642 | 2.530 | 90.3 | 84.6 |
| DFlash2 K=7 | 3.151 | 3.048 | 84.9 | 80.1 |
| MTP K=3 | 2.664 | **2.716** | 91.5 | 90.9 |
| MTP K=7 | 3.208 | **3.275** | 82.2 | 82.0 |

DFlash2 is worse on prose than code at both widths; MTP is *better* on prose
than code at both widths. A sign flip, not a magnitude difference, and
consistent across K -- DFlash2's lattice appears tuned to something
code-shaped. It is also why DFlash2's running acceptance drifted down when
the prose half of the corpus began while MTP's held flat.

### 4. Per-position acceptance

```
DFlash2 K=3   72.0  50.8  35.6
DFlash2 K=7   69.8  47.8  32.7  22.7  16.3  11.8   8.8
MTP K=3       74.5  54.4  40.2
MTP K=7       71.2  49.0  34.6  25.0  18.6  14.4  11.5
```

Position 6 fires on under 9% of DFlash2 steps. But the K=3/K=7 comparison
shows the cost is in *offering* the positions, not their yield: removing
them saves ~21-25% of step time, which is the whole win.

### 5. Prefill dominates end to end

TTFT is ~23s against ~22-25s of decode, so **~50% of every request is
prefill** and all four arms land within 6.7% end-to-end (44.9-47.9 s/req).
For a user of this deployment the speculator choice is nearly irrelevant;
the 13.9% prefix-cache hit rate measured on the live server is worth far
more than any of these four.

## Confounds and limits, stated plainly

**Expert residency is not equal.** MTP runs 2325 hot experts to DFlash2's
2224 -- 101 more, because the planner correctly charges DFlash2 for its
2.289 GiB draft cache (2.289 GiB / 20.3 MiB = 113, as predicted). This is a
genuine deployment cost of a dense drafter and belongs in the comparison,
but part of MTP's throughput lead is a lower cold-expert hit rate rather
than better drafting. **The acceptance numbers are clean of this; the tok/s
numbers are not.**

**Nearly every request was truncated mid-reasoning.** 55-57 of 59 requests
per arm never closed `</think>` at the 2048-token cap. GLM-5.3 reasons
before answering and these tasks are long. So this is a *reasoning-phase*
comparison. The speculator ranking is sound -- same corpus, same cap, same
everything -- and the code/prose split is real, but it is a split in how the
model reasons about code versus prose, not in finished code versus finished
prose. Settling that needs a rerun at 4096.

**The T=0 probes do not prove losslessness.** Probe 0 (`391`) and probe 3's
answer (`probe ok 4`) matched across all four arms, but the reasoning traces
diverged. That is expected floating-point non-determinism -- different K
means different verify batch shapes, different kernels, different reduction
orders, and near-tied logits then resolve differently -- not a speculative
decoding bug. The check as designed cannot separate the two, so it should be
read as evidence that DFlash2 at K=3, an untrained width for a checkpoint
whose `dflash_config` declares `block_size: 8`, produces coherent on-topic
output rather than garbage. It does.

**No reasoning parser.** The arms serve without `--reasoning-parser`, unlike
the production launcher, so thinking stays inline in `content`. Generation is
identical; only API-level splitting differs.

## Recommendation

**Serve Claude Code with MTP at K=3.** Best decode throughput (91.2 tok/s),
best acceptance at that width, 101 more resident experts, no drafter
checkpoint to load, and no dense-drafter KV to budget -- which is also the
whole class of planner bug that cost this campaign four failed launches.

DFlash2's advantage on this fork was assumed from a GSM8K number that does
not survive contact with agentic traffic.

## Files

- `build_prompts.py` / `corpus.jsonl` -- corpus builder and the 60 frozen requests
- `arm.sh` / `submit-all.sh` -- one arm, and the four-job submitter
- `bench_client.py` -- streaming client, per-request `/metrics` deltas
- `aggregate.py` -- the tables above, from the four `*-result.json`
