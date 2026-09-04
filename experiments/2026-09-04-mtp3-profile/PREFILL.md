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
Marlin call uses the default legacy launch. And **cold costs 2.1x hot** despite
owning fewer experts (20 vs 44).

## Roofline

Ceilings are GH200 120GB **spec**. The project's earlier roofline used
conservative values (HBM 3.5 TB/s, BF16 630 TF/s, FP8 1260); against those,
five rows here exceed 100%, which is a statement about the ceiling, not the
kernel. A row's roof is `min(compute ceiling, AI x bandwidth)`.

| role | ms | % | AI | achieved | eff BW | roof | eff% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse MLA FP8, **64 heads launched** | 543.8 | 27.7 | 212 | 348 TF/s | 1639 G | 849 T | 41.0 |
| &nbsp;&nbsp;*the 16 real TP heads only* | 543.8 | 27.7 | 53 | 87 TF/s | | 849 T | **10.2** |
| Marlin W4 cold w13 | 385.3 | 19.6 | 664 | 63 TF/s | 95 G | 248 T | 25.4 |
| Marlin W4 cold w2 | 195.1 | 9.9 | 572 | 62 TF/s | 109 G | 213 T | 29.2 |
| Marlin W4 hot w13 | 127.7 | 6.5 | 664 | 294 TF/s | 443 G | 989 T | 29.8 |
| Marlin W4 hot w2 | 70.0 | 3.6 | 572 | 269 TF/s | 470 G | 989 T | 27.2 |
| DSA indexer FP8 scan | 95.1 | 4.8 | 1970 | 486 TF/s | 247 G | 1979 T | 24.5 |
| BF16 GEMM o_proj 4096x6144 | 53.6 | 2.7 | 1890 | 623 TF/s | 329 G | 989 T | 63.0 |
| BF16 GEMM QKV-A 6144x2624 | 33.6 | 1.7 | 1502 | 637 TF/s | 424 G | 989 T | 64.4 |
| BF16 GEMM q_b 2048x4096 | 21.7 | 1.1 | 1170 | 664 TF/s | 567 G | 989 T | 67.1 |
| BF16 GEMM 6144x1024 | 10.8 | 0.5 | 793 | 745 TF/s | 940 G | 989 T | 75.3 |
| BF16 MLA W_UK/W_UV bmm | 10.8 | 0.5 | 137 | 388 TF/s | 2824 G | 549 T | 70.6 |
| MoE `silu_and_mul` | 32.8 | 1.7 | -- | -- | 3752 G | 4000 G | 93.8 |
| DSA top-k per row | 33.7 | 1.7 | -- | -- | 1364 G | 4000 G | 34.1 |
| TP all-reduce (bus bytes) | 195.6 | 10.0 | -- | -- | 123 G | 450 G | 27.4 |

Spec: HBM 4.0 TB/s | C2C 373 GB/s measured (450 spec) | NVLink one-way
aggregate 3x150 = 450 GB/s | BF16 989 TF/s | FP8 1979 TF/s.

**The dense BF16 GEMMs are fine** (63-75% of peak) and `silu_and_mul` is at
94% of HBM. Nothing to win there. Everything below 30% is the story.

### Hot and cold Marlin are neither compute- nor bandwidth-bound

Every Marlin launch, hot or cold, uses `grid=(264,1,1) block=(128,1,1)
smem=115200` -- 132 SMs x 2 blocks with the legacy half-SM shared-memory
request. The grid never changes with expert count or M.

Hot sits at 30% of the BF16 roof and 443 GB/s of a 4 TB/s HBM; cold at 25% of
its C2C-limited roof and 95 GB/s of 373 GB/s. The rank-1 clock difference
(below) settles which: **Marlin is clock-insensitive** (+1.2% cold, +1.8% hot,
against +8.5% for a genuinely compute-bound GEMM), so it is not SM-bound, and
it is not at either bandwidth ceiling either. `2026-07-29-marlin-smem-monopoly`
hit the same wall from the other direction -- "whatever holds W4A16 Marlin to
~11% of peak is not reachable from the launch configuration".

One caveat on cold's roof: if Marlin re-reads each expert's weights once per
64-row M-block (~256 rows/cold expert -> ~4 blocks), physical AI is ~166 and
the C2C roof falls to ~62 TF/s -- exactly the 63 measured. The trace cannot
confirm the re-read factor because the grid is fixed, so cold being *at* the
C2C ceiling remains a live alternative to it being 25% below it. That changes
the fix (fewer cold bytes vs. a better kernel) but not the 580 ms size.

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
2. **Cold tier costs 580 ms, 29.5%, for 20 of 64 experts.** 2.1x hot's time
   for fewer experts. Either it is at the C2C ceiling with M-block weight
   re-reads (then the fix is fewer cold bytes -- promotion, or an M-tiling
   change) or it is 25% below it (then the fix is the kernel). Resolve the
   re-read factor first; the two fixes are unrelated.
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
