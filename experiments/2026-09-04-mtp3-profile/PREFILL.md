# The prefill chunk, kernel by kernel

MTP3 / c=1 / DCP1 / 400K, job 1665068, GLM-5.3 W4A16 tiered, TP4+EP4 on one
GH200 node. One chunk is **8192 tokens** at ~96K context; a request is ~12 of
them. Rank 0, mean of the 6 captured chunks.

Every kernel here is joined to the CPU operator that launched it through the
profiler's `External id`, so operand shapes are **observed**, not modelled.
FLOPs and logical bytes are derived from those shapes plus the config; they are
not measured DRAM transactions.

## At a glance

| | ms |
| --- | ---: |
| wall (engine annotation to annotation) | 1902.5 |
| GPU span | 1974.6 |
| GPU busy (union of kernel intervals) | 1964.6 |
| cumulative kernel time | 1964.8 |

`cumulative / busy = 1.000`: **prefill is single-stream**, so every share below
is additive. GPU busy counts NCCL spin-wait as busy; compute-only is ~1865 ms.
Percentages are of the 1964.6 ms busy figure.

## Where the time goes

| operator | tier/shape | calls | ms | % | operands |
| --- | --- | ---: | ---: | ---: | --- |
| `sparse_decode_fwd` | | 81 | 543.8 | 27.7 | q`[1,8192,64,576]` idx`[1,8192,2048]` |
| `moe_wna16_marlin_gemm` | **cold** w13 | 75 | 385.3 | 19.6 | `[20,384,8192]` int32 |
| `all_reduce` | | 160 | 195.6 | 10.0 | `[8192,6144]` bf16 |
| `moe_wna16_marlin_gemm` | **cold** w2 | 75 | 195.1 | 9.9 | `[20,128,12288]` |
| `moe_wna16_marlin_gemm` | hot w13 | 75 | 127.7 | 6.5 | `[44,384,8192]` |
| `sparse_attn_indexer` | | 24 | 95.1 | 4.8 | q`[8192,32,128]` fp8 |
| `moe_wna16_marlin_gemm` | hot w2 | 75 | 70.0 | 3.6 | `[44,128,12288]` |
| `aten::mm` o_proj | 4096x6144 | 81 | 53.6 | 2.7 | `[8192,4096]x[4096,6144]` |
| `top_k_per_row_prefill` | | 81 | 33.7 | 1.7 | `[4096,32768]` f32 |
| `aten::mm` QKV-A | 6144x2624 | 81 | 33.6 | 1.7 | `[8192,6144]x[6144,2624]` |
| `silu_and_mul` | | 153 | 32.8 | 1.7 | `[65536,4096]` |
| `aten::copy_` | | 263 | 23.4 | 1.2 | MLA head padding |
| `aten::mm` q_b | 2048x4096 | 105 | 21.7 | 1.1 | `[8192,2048]x[2048,4096]` |
| `aten::fill_` | | 269 | 14.6 | 0.7 | MLA head padding |
| 55 smaller roles | | | 138.1 | 7.0 | |
| **total** | | | **1964.8** | **100.0** | |

Rolled up: **attention 32.5%** (MLA + indexer + top-k + MLA GEMMs),
**routed MoE 39.6%**, **communication 10.0%**, dense/shared GEMMs 7.8%, glue 10%.

## One layer, in launch order

From chunk 3, between two all-reduces. Sub-totals per layer:

| phase | ms | what runs |
| --- | ---: | --- |
| attention | 9.07 | RMS -> QKV-A -> q_b -> rope/cache -> W_UK bmm -> **sparse MLA 6.69** -> W_UV -> o_proj |
| all-reduce (post-attention) | 1.08 | |
| MoE | 12.19 | RMS -> shared expert -> topk/align -> **hot w13 2.22 + w2 1.21** -> **cold w13 4.87 + w2 2.46** |
| all-reduce (post-MoE) | 0.46 | |
| **per layer** | **22.80** | x78 layers + embed/vocab = 1965 ms |

Two things this shows. **Hot and cold tiers are strictly serial** (2.22+1.21
then 4.87+2.46) -- prefill never forks the tier streams, because the launch
policy is bypassed at M=8192 (`launch_policy.max_tokens >= M` fails), so every
Marlin call uses the default legacy launch. And **cold costs 2.94x hot's time
for slightly more experts**: 580.4 ms over 2475 experts against 197.7 ms over
2325, which is 2.8x per expert.

Those counts are measured two independent ways that agree exactly. The server
log reports `2325 hot / 2475 cold experts per rank (46.0 GiB available /
20.3 MiB per expert)`, and summing the expert dimension of every Marlin operand
across the 75 layer pairs gives the same 2325 / 2475. Note the runtime
**demotes** the placement profile's 2496 hot experts to 2325 to fit the 46 GiB
of HBM it actually has, which is why the profile's per-layer hot counts do not
match the trace. Hot averages 31.0 experts per layer and cold 33.0, and hot
thins out with depth (33.2 over layers 0-36, 28.9 over 37-74) as the promotion
budget runs out.

## Roofline

Ceilings are GH200 120GB **spec**. The project's earlier roofline used
conservative values (HBM 3.5 TB/s, BF16 630 TF/s, FP8 1260); against those,
five rows here exceed 100%, which is a statement about the ceiling, not the
kernel. A row's roof is `min(compute ceiling, AI x bandwidth)`.

| role | ms | % | AI | achieved | eff BW | roof | eff% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse MLA FP8, **64 heads launched** | 543.8 | 27.7 | 212 | 348 TF/s | 1639 G | 849 T | 41.0 |
| &nbsp;&nbsp;*the 16 real TP heads only* | 543.8 | 27.7 | 53 | 87 TF/s | | 849 T | **10.2** |
| Marlin W4 cold w13 | 385.3 | 19.6 | 664 | 83 TF/s | 125 G | 248 T | 33.4 |
| Marlin W4 cold w2 | 195.1 | 9.9 | 572 | 82 TF/s | 143 G | 213 T | 38.3 |
| Marlin W4 hot w13 | 127.7 | 6.5 | 664 | 235 TF/s | 353 G | 989 T | 23.7 |
| Marlin W4 hot w2 | 70.0 | 3.6 | 572 | 214 TF/s | 374 G | 989 T | 21.6 |
| DSA indexer FP8 scan | 95.1 | 4.8 | 1970 | 486 TF/s | 247 G | 1979 T | 24.5 |
| BF16 GEMM o_proj 4096x6144 | 53.6 | 2.7 | 1890 | 623 TF/s | 329 G | 989 T | 63.0 |
| BF16 GEMM QKV-A 6144x2624 | 33.6 | 1.7 | 1502 | 637 TF/s | 424 G | 989 T | 64.4 |
| BF16 GEMM q_b 2048x4096 | 21.7 | 1.1 | 1170 | 664 TF/s | 567 G | 989 T | 67.1 |
| BF16 GEMM 6144x1024 | 10.8 | 0.5 | 793 | 745 TF/s | 940 G | 989 T | 75.3 |
| BF16 MLA W_UK/W_UV bmm | 10.8 | 0.5 | 137 | 388 TF/s | 2824 G | 549 T | 70.6 |
| MoE `silu_and_mul` | 32.8 | 1.7 | -- | -- | 3752 G | 4000 G | 93.8 |
| DSA top-k per row | 33.7 | 1.7 | -- | -- | 1364 G | 4000 G | 34.1 |
| TP all-reduce (bus bytes) | 195.6 | 10.0 | -- | -- | 123 G | 450 G | 27.4 |

Ceilings, from the [JSC JUPITER configuration
page](https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html) except
where noted:

| resource | ceiling | source |
| --- | ---: | --- |
| HBM3 | 4.0 TB/s (96 GB) | JSC |
| Grace LPDDR5X | 512 GB/s (120 GB) | JSC |
| NVLink-C2C, CPU-GPU | 900 GB/s -> **450 GB/s per direction** | JSC |
| &nbsp;&nbsp;measured achievable read | 373 GB/s | `2026-07-25-grace-bandwidth` |
| NVLink-4 GPU-GPU | 300 GB/s per pair, 150 per direction; 3 peers -> **450 GB/s egress** | JSC |
| BF16 / FP8 tensor cores | 989 / 1979 TF/s dense | H100-SXM spec; 132 SMs confirmed by JSC and the trace |

The JSC page replaces the 421 GB/s C2C figure this project had carried since
July: the link is 450 GB/s per direction and Grace's own memory is 512 GB/s, so
a GPU reading Grace is bounded by 450. Cold rows are scored against the 373
GB/s *measured* rate; against the 450 spec they fall to 27.7% / 31.7%.

The 450 GB/s NVLink egress figure is independently corroborated by the trace:
310 GB/s of bus bandwidth exceeds any single 150 GB/s pair link, so NCCL is
running multiple rings, and 3x150 is the right aggregate.

**The dense BF16 GEMMs are fine** (63-75% of peak) and `silu_and_mul` is at
94% of HBM. Nothing to win there. Everything below 30% is the story.

### Hot is bound by neither ceiling; cold plausibly sits on C2C

Every Marlin launch, hot or cold, uses `grid=(264,1,1) block=(128,1,1)
smem=115200` -- 132 SMs x 2 blocks with the legacy half-SM shared-memory
request. The grid never changes with expert count or M.

Weight traffic per chunk, logical (each expert read once):

| tier | experts | weight bytes | ms | logical BW | x4 M-block re-read | vs ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hot (HBM) | 2325 | 49.4 GB | 197.7 | 250 GB/s | 999 GB/s | **25% of 4.0 TB/s** |
| cold (Grace) | 2475 | 52.6 GB | 580.4 | 91 GB/s | **362 GB/s** | **97% of 373 GB/s measured** |

Cold streams essentially the whole cold expert set once per chunk -- at M=8192
every owned expert is activated. Each cold expert takes ~256 routed rows, so if
Marlin re-reads an expert's weights once per 64-row M-block, physical traffic is
~4x logical: **362 GB/s against the 373 GB/s measured C2C read rate, a 97% fit**
(81% of the 450 GB/s spec). That is a good fit but not a measurement -- the grid
is fixed, so the trace cannot show the re-read factor, and the alternative is
that cold is simply 3-4x off its ceiling.

Hot admits no such reading: even at 4x re-read it is at 25% of HBM, and it runs
at 23.7% of the BF16 roof. The rank-1 clock difference (below) settles that it
is not SM-bound either -- **Marlin is clock-insensitive** (+1.2% cold, +1.8%
hot, against +8.5% for a genuinely compute-bound GEMM).
`2026-07-29-marlin-smem-monopoly` hit the same wall from the other direction:
"whatever holds W4A16 Marlin to ~11% of peak is not reachable from the launch
configuration". That dead end stands for hot; cold now has a candidate answer.

## Communication: latency, bandwidth, and the actual bottleneck

160 all-reduces per chunk, all one shape: `[8192, 6144]` bf16 = **100.7 MB**.
Ring bus bytes per GPU = `2(N-1)/N x operand` = 151 MB.

| | calls | transfer | wait | total | % wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| post-MoE all-reduce | 77 | 36.87 ms | **78.78 ms** | 115.66 ms | 68.1% |
| post-attention all-reduce | 82 | 40.13 ms | 21.45 ms | 61.57 ms | 34.8% |
| **total** | 160 | **77.5 ms** | **100.2 ms** | 177.7 ms | 56.4% |

Transfer is split by taking, for each (chunk, ordinal), the **minimum across
the four ranks** -- a collective that is genuinely slow is slow on all four at
once, and that never happens (0 of 960).

- **Latency** is not the limit: 100.7 MB is far past any latency regime.
- **Bandwidth** is not the limit: at the cross-rank minimum (487 us) the ring
  runs at **310 GB/s bus = 69% of the 450 GB/s NVLink one-way aggregate**. The
  177.7 ms *mean* implies 123 GB/s, but that number is measuring waiting.
- **The bottleneck is arrival skew**, and it is 56% of all communication time.

Both transfer figures are equal (36.9 vs 40.1 ms) because the bytes are
identical; only the wait differs. A footnote worth chasing separately: the
kernel is `RING_**LL**`, NCCL's low-latency protocol, for a 100 MB operand --
LL carries flags inline and roughly doubles wire bytes, and the tuner normally
switches to Simple far below this size. No `NCCL_*` variable is set in
`jupiter-env.sh`, `run-server.sh` or the job script, but the module-loaded
environment and `/etc/nccl.conf` were not checked.

## What the skew actually is

Per layer, comparing the slowest rank against the mean of the four:

| segment | mean rank | slowest rank | excess | matches |
| --- | ---: | ---: | ---: | --- |
| routed MoE Marlin | 783.07 ms | 861.31 ms | **78.24 ms (10.0%)** | post-MoE wait 78.78 ms |
| sparse MLA | 550.71 ms | 562.90 ms | 12.20 ms (2.2%) | post-attention wait 21.45 ms |

The MoE excess reproduces the post-MoE all-reduce wait to within 0.7%. **The
dominant skew is per-layer routed-expert load imbalance** -- each rank owns 64
experts but the tokens routed to them, and the hot/cold split, differ layer by
layer, so the slowest rank changes from layer to layer.

### A second, smaller effect: rank 1 runs slower

Rank 1 holds the cross-rank minimum 58.8% of the time (25% = chance), i.e. it
arrives last. Per kernel, rank 1 minus the mean of the others:

| kernel | r1 - mean | relative | bound by |
| --- | ---: | ---: | --- |
| `aten::mm` o_proj | +4.61 ms | **+8.5%** | compute |
| `sparse_attn_indexer` | +5.02 ms | +5.2% | compute |
| `sparse_decode_fwd` | +11.78 ms | +2.1% | gather |
| Marlin hot w13 | +2.27 ms | +1.8% | neither |
| Marlin cold w13 | +4.46 ms | +1.2% | C2C |
| `silu_and_mul` | +0.31 ms | +0.9% | HBM |
| `aten::fill_` | +0.01 ms | +0.07% | HBM |

The slowdown scales with how compute-bound a kernel is and vanishes for
memory-bound ones. That is the signature of a **lower SM clock on GPU 1**, not
a memory-placement problem. Positive deltas sum to ~36.9 ms, against the
40.7 ms *less* that rank 1 waits in all-reduce -- it closes. GPU busy per chunk
is equal across ranks (1964.6 / 1964.8 / 1965.0 / 1965.5), which is what makes
the barrier absorb it. Test: `nvidia-smi -q -d CLOCK,PERFORMANCE -i 1` on the
node under load.

## Sampling: what grows with context

These are chunks 1-6 of ~12, so anything that scans preceding KV is understated.

| role | c1 | c2 | c3 | c4 | c5 | c6 | shape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sparse_attn_indexer` | 51.8 | 72.7 | 93.8 | 104.9 | 111.0 | 136.3 | **linear in context** |
| `top_k_per_row_prefill` | 22.2 | 27.1 | 31.7 | 35.3 | 40.7 | 45.4 | **linear** |
| `sparse_decode_fwd` | 541.4 | 540.9 | 545.4 | 543.1 | 547.4 | 544.7 | flat (top-2048) |
| `all_reduce` | 193.2 | 190.7 | 190.2 | 194.3 | 204.6 | 200.9 | flat |
| Marlin hot w13 | 124.0 | 131.8 | 128.3 | 130.2 | 125.3 | 126.8 | flat |

Extrapolated linearly to chunk 12, the indexer reaches ~270 ms and top-k ~90 ms
per chunk, so over a **full** 96K prefill the DSA index path is roughly double
its 6.5% here -- about 13% -- while MLA, Marlin and the all-reduce stay put.

## Ranked levers

1. **Sparse MLA launches 64 heads for 16 real ones -- 543.8 ms, 27.7%.** The
   rank owns `64/TP4 = 16` query heads, confirmed by every neighbouring op
   (`concat_mla_q [8192,16,512]`, `bmm [16,8192,192]`), but FlashMLA's sparse
   kernel supports 64 or 128, so vLLM launches 64. Three quarters of the work
   is padding: 41.0% of the FP8 roof as launched, **10.2% useful**. Add the
   padding machinery immediately before it -- `fill_` 178 us + `copy_` 187 us
   per layer, ~38 ms/chunk -- and the lever is **~446 ms/chunk, 22.7%**. Needs
   a 16-head kernel, not a flag.
2. **Cold tier costs 580 ms, 29.5%, and it is probably at the C2C ceiling.**
   2475 experts against hot's 2325 -- *more* experts, 2.94x the time, 2.8x per
   expert. At a 4x M-block weight re-read it runs at 362 GB/s against the
   373 GB/s measured C2C read rate, a 97% fit. If that is right the lever is
   the re-read, not the link: removing it would take cold from 580 ms toward
   ~145 ms, **~435 ms/chunk, 22%** -- comparable to the MLA lever, and it needs
   Marlin restructured to loop M inside a resident weight tile. Confirm the
   re-read factor first; it is inferred, not measured. Promotion is not the
   answer -- HBM is already full at 46.0 GiB, and the runtime already demotes
   the profile's 2496 hot experts to 2325.
3. **Arrival skew -- 100.2 ms/chunk, 5.1%.** 78.8 ms of it is routed-expert
   imbalance across ranks, 21.5 ms is attention. Not a comms fix.
4. **Rank 1's clock -- worth ~37 ms/chunk** if it is a recoverable power or
   thermal cap. One `nvidia-smi` check.
5. **The DSA index path doubles over a full prefill** (~13% at chunk 12).
   `index_topk_freq=4` already limits the full scan to every fourth layer;
   `top_k_per_row_prefill` still runs on all 81.
6. **Prefill Marlin uses the legacy launch.** `e59d34275`'s `(64,256)` tile,
   measured 1.10-1.12x faster on exactly this configuration, is **not in this
   tree**, and the launch policy is bypassed at M=8192 anyway. Worth ~1% of
   prefill; small, but it is a known-good change.

---

# Can a cold-expert H2D prefetch be fully hidden?

**Yes, with 7x margin in the worst case, and the buffer fits.** Measured, not
modelled: for all 444 layer transitions in the 6 captured chunks, the window
between "layer L's cold buffer is free" and "layer L+1's cold Marlin starts"
against the H2D time its operands actually need.

| | min | p5 | median | max |
| --- | ---: | ---: | ---: | ---: |
| window (end of cold L -> start of cold L+1) | 12.96 | 13.70 | 16.11 | 26.90 ms |
| H2D needed @450 GB/s | 0.85 | | 1.65 | 1.93 ms |
| **slack** | **11.12** | | 14.53 | | ms |
| **window / need** | **7.04x** | | 10.28x | | |

**0 of 444 transitions fail at 450 GB/s, and 0 of 444 fail even at the 373 GB/s
measured rate.** Aggregate: 52.13 GB of cold weights per chunk = 115.8 ms of
DMA against a 1903 ms chunk, a **6.1% duty cycle**.

One structural property makes this robust: the window *excludes* cold Marlin by
construction, so it is made of attention, the two all-reduces and the hot tier.
Speeding cold up does not shrink its own prefetch window.

## The buffer, not the bandwidth, is the constraint -- and it fits

| | |
| --- | ---: |
| largest single layer | 830 MiB (41 cold experts) |
| double-buffered | **1.62 GiB** |
| observed free HBM | 10.33 GiB |
| planner minimum | 8.38 GiB |
| **margin** | **1.95 GiB** |

1.62 fits inside 1.95, with 0.33 GiB to spare. If that is too tight, demoting
16 hot experts (20.3 MiB each) buys another 0.33 GiB and costs nothing once
cold runs at hot's speed. Single-buffering -- the zero-then-refill scheme --
halves it to 0.83 GiB, and the windows above are already measured for exactly
that scheme (the copy starts when the buffer is *consumed*, at the end of cold
L). Double-buffering would let the copy start a full layer earlier still.

**Drop the zeroing step.** The buffer is entirely overwritten by the incoming
H2D, so clearing it first buys nothing and costs 0.17-0.21 ms per layer of HBM
write bandwidth on the same HBM the compute is using.

## What it is worth

The saving rests on cold-in-HBM running at hot's rate. The evidence for that is
direct: **hot and cold are the same kernel, same grid `(264,1,1)`, same operand
shapes, same M-tiling -- the only variable is where the weights live, and cold
is 2.8x slower per expert.** That argument does not depend on the 4x M-block
re-read hypothesis.

| | now | with prefetch |
| --- | ---: | ---: |
| cold Marlin | 583.3 ms | 210.3 ms |
| **saving** | | **372.9 ms/chunk, 19.0%** |

Treat 373 ms as an **upper bound**. Cold experts are cold precisely because
they take fewer tokens, so their rows-per-expert is lower than hot's and Marlin
would get worse M-utilisation from the same weights; the projection assumes
uniform routing, under which both tiers average ~256 rows per expert. The real
split is skewed toward hot.

## Why this is prefill-only, and why it does not replace tiering

Not a bandwidth argument -- a **predictability** one. At M=8192 every owned
expert fires, so "which cold experts does layer L+1 need" has the answer "all
of them" before layer L even runs. At decode's M=4 roughly one cold expert per
rank per layer fires, and which one is unknown until the router runs *for that
layer*. There is nothing to prefetch. Decode's cold path is also already at the
C2C roofline (53 us/layer for ~20.1 MB = 379 GB/s, `2026-07-29`), so there is
no headroom there either. **The 46 GiB of hot residency stays; this adds a
prefill path beside it.**

## It composes with the MLA lever

A 16-head MLA kernel removes ~5.0 ms/layer from attention, which sits inside
the prefetch window. Window median falls 16.11 -> ~11.1 ms and p5 13.70 -> ~8.7
ms, against an unchanged 1.65 ms median need: still 5-6.7x margin.

| | ms/chunk | TTFT at 96K |
| --- | ---: | ---: |
| today | 1903 | 23.2 s |
| + cold prefetch | ~1530 | ~18.7 s |
| + 16-head MLA | ~1084 | **~13.2 s** |
