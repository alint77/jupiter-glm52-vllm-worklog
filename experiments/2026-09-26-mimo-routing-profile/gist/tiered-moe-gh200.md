# Tiered MoE on GH200: serving models bigger than HBM

How we serve MoE models that don't fit in HBM on one 4x GH200 node with vLLM.
Keep the experts that actually get used in HBM, leave the rest in Grace memory,
and have the GPU read **both tiers at the same time**. Making them overlap was
the core of the work; everything else builds on it.

**TL;DR**

- GH200's GPU can read Grace memory in place at ~410 GB/s, 1/9 of HBM. That makes
  Grace usable for weights, but only if Grace reads hide *under* HBM reads instead
  of adding to them.
- We split every MoE layer per expert into a hot tier (HBM) and a cold tier
  (Grace) and run both at once. That took fixing a Marlin launch detail that
  silently serialized the two kernels.
- GLM-5.2 (361 GiB) at 400K context: stock vLLM offload decodes at **38 tok/s**,
  our stack at **128 tok/s**. Overlap is +18% of that on its own, plus another
  −8% step time from the Marlin fix.
- On top of that, picking the hot set from real routing traces cut MiMo-V2.6's
  decode step by **28%**.

## The hardware

```mermaid
flowchart LR
  HBM["HBM3, 96 GB<br/>~3.6 TB/s"] --- GPU["Hopper GPU"]
  GPU ---|"NVLink-C2C<br/>~410 GB/s GPU reads"| Grace["Grace CPU<br/>LPDDR5X, ~120 GB"]
```

A GPU kernel can dereference Grace memory directly (UVA): no copy, no page fault.
That's ~120 GB more per GPU at 1/9 of HBM speed. All numbers here are measured
on our node, not spec sheets.

## The problem

- **GLM-5.2 W4A16** is 361 GiB of weights against 4 x 96 GiB of HBM on the node, before
  any KV cache.
- **MiMo-V2.6-Pro** has 6,624 experts per GPU at EP4: 123.7 GiB of expert weights
  for 95 GiB of HBM that also holds attention weights, a 250K-token KV cache and a
  speculative drafter.

So 35–50% of experts have to live in Grace, depending on model and context. MoE makes that workable, because a
step only touches a few experts per layer. The question is what the Grace part
costs.

## What stock vLLM does

`--cpu-offload-gb N` moves parameters to pinned host memory, in model order, until
it has moved N GB. The GPU then reads them in place through UVA. On GH200 that's
the right mechanism with the wrong shape:

- **Whole layers go to Grace.** Busy and idle experts are treated the same.
- **A layer lives in exactly one tier.** While a Grace-resident layer runs, HBM
  sits idle, and vice versa. Grace time is *added* to the step, never hidden.
- **Sharp edges:**
  - The flag is silently ignored under the V2 model runner.
  - `pin_memory()` rounds every allocation up to a power of two, so a 1.01 GiB
    tensor pins 2 GiB.
  - Pinned pages land on whatever NUMA node the thread is on. The wrong node
    drops C2C from ~410 to 70–80 GB/s, and nothing errors.

The prefetch offloader copies whole layers into HBM ahead of use. For MoE decode
that moves every expert a GPU owns in a layer (64–96) to use the ~11 a step
touches,
and its staging buffers take HBM that could hold experts.

## The idea: two tiers, read at the same time

Split each MoE layer's experts into **hot** (HBM) and **cold** (Grace). After
routing, launch one Marlin kernel over the hot experts and one over the cold
experts, on two streams, so the layer costs `max(hot, cold)` instead of
`hot + cold`.

![concept](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/overlap-concept.png)

If the tiers run one after the other, every byte from Grace costs 9x a byte from
HBM, and offloading is pure loss. If they overlap, Grace reads hide under HBM
reads until both take equally long, which happens at ~16% of a step's bytes from
Grace. Up to there, offloading is close to free.

## Making the overlap real

**The first version didn't overlap.** It had per-expert tiers, two launches and
two streams, and still decoded at stock-offload speed (37 vs 38 tok/s): the tiers
ran effectively back to back. Enabling the overlap for the 4-token MTP verify
batches brought +18% (108 → 128 tok/s). Profiling then showed the kernels still
barely overlapped: 103 µs together against 113 µs one after the other.

**The cause was Marlin's shared-memory request.** Marlin sizes its launch to
carve each SM into exactly N CTAs, so it asks for `228 KiB / N` of shared memory
per CTA regardless of what it uses. It uses ~25 KiB. At 3 CTAs per SM, the hot
kernel claims 224 of 228 KiB on every SM, and the cold kernel's CTAs can't be
placed anywhere until hot CTAs retire.

![smem](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/overlap-smem.png)

**The fix:** request the shared memory the kernel actually indexes, and size the
grid explicitly: hot at 2 CTAs per SM, cold at 1, both spread over all 132 SMs.
Now CTAs from both tiers sit on the same SMs, one kernel pulling from HBM and the
other from Grace.

![sweep](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/overlap-sweep.png)

Across 15 realistic hot/cold mixes, one layer gets 15–40% faster (median −34%),
and most land within a few µs of `max(hot, cold)`. End to end on GLM, same node,
acceptance-adjusted: **−7.7% step time, +7% tok/s**, and +6% on long agentic
prompts. Three things we checked along the way:

- **HBM and C2C traffic don't fight.** 3.0 TB/s from HBM and 410 GB/s over C2C,
  co-resident, finished within 0.1% of the slower one alone.
- **The cold kernel saturates the link.** Cold Marlin reaches 88–95% of C2C.
- **Partitioning SMs is worse.** Green contexts, which give each tier its own
  SMs, were 9–40% slower. With 8 SMs the cold tier is 2.5x slower than with 32:
  it needs many SMs to keep enough reads in flight.

![ladder](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/ladder.png)

Blue bars are offload work; grey bars are a MoE communication fix and MTP
speculative decoding.

## Making it production-grade

- **Exact memory planning.** A planner counts every byte on the GPU: weights, KV
  cache, workspaces, CUDA graphs and a reserve. Whatever is left becomes hot
  expert slots. Free HBM after warmup matches the plan to ~0.05 GiB.
- **Load straight to the final tier.** Each expert is converted to Marlin layout
  one at a time and written directly to HBM or Grace. The cold tier is pinned at
  exact size (`cudaHostRegister`, no rounding) on the GPU-local NUMA node.

## Choosing what's hot

Once the tiers overlap, the goal is to keep the cold tier's share of each step
near the ~16% balance point. Routing is skewed: some experts get ~8x a uniform
share, and the bottom ~50 per layer are almost never picked. So we record which
experts the router picks on real agentic-coding sessions, and fill the hot slots
with the most-used ones.

![coverage](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/2-cumulative-hbm-share.png)

Measured on task types the ranking never saw: with the most-used half in HBM,
80% of routed tokens never touch Grace. With an arbitrary half it's 50%.

MiMo had been running with an arbitrary hot set: 42% of each step's expert reads
came from Grace, far past the balance point. With the profile it's 17%.

![decode ab](https://gist.githubusercontent.com/alint77/d8b18397c1f8444544f2425910b7fc7b/raw/5-decode-ab.png)

MiMo-V2.6-Pro, 250K context, batch one, DFlash speculative decoding (8 tokens
verified per step). Only the hot set changes; servers are restarted per run, in
both orders, on three nodes. The decode prompts weren't in the capture.

| | arbitrary hot set | routing profile |
| --- | ---: | ---: |
| expert reads from Grace per step | 42% | **17%** |
| decode step time | 34.7 ms | **25.0 ms (−28%)** |
| decode speed | ~110 tok/s | **~147 tok/s** |
| TTFT at 32K / 128K / 240K | 4.0 / 16.9 / 40.2 s | 4.0 / 16.7 / 39.8 s |
| GSM8K 400, two runs each | 90.2 / 91.5% | 90.7 / 90.0% |

For GLM at 4 concurrent requests we also keep second copies of busy cold experts
on other GPUs and route each token to whichever copy balances the ranks:
another −5 to −6.5% step time.

## Prefill is different

A prefill chunk is 8K tokens, so every cold expert gets hit many times. GPU L2
doesn't cache host memory, so Marlin re-streams the same cold weights over C2C
for every token block. In prefill we instead copy the next layer's cold experts
into an HBM staging slot while the current layer computes: **−17 to −23% TTFT**
on MiMo. Decode keeps reading in place.

## What each piece bought

Each result is against its own matched control.

| change | effect |
| --- | --- |
| hot/cold overlap on two streams (MTP3 verify) | +18% decode tok/s (GLM) |
| Marlin shared-memory fix, tiers truly co-resident | −34% per MoE layer, −7.7% step (GLM) |
| hot set from routing traces | −28% decode step (MiMo) |
| cross-GPU copies of busy cold experts | −5 to −6.5% step at 4 concurrent requests (GLM) |
| staging cold experts in prefill | −17 to −23% TTFT (MiMo) |

## Things worth knowing

- **Offloading barely matters at one token per step.** There, HBM-resident Marlin
  is latency-bound (~10% of HBM bandwidth), and Grace-resident experts ran within
  a few percent of it. With speculative decoding or batching, a step touches ~45 of
  384 experts per layer, kernels turn bandwidth-bound and the 9x gap shows. That's
  when both overlap and placement start paying.
- **Measure under CUDA graphs.** Timing the fork/join eagerly added ~110 µs of
  stream barriers per iteration. That hid the whole overlap win and made the fix
  look worthless at low expert counts.
- **Profiles are traffic-specific.** Ours is trained on coding. It holds up on
  coding task types it never saw (15% vs 12% of routes to Grace in-sample), but
  chat, math or other languages are untested.
