# Full-footprint host-UVA MLA cache gate

Phase 5 cache-tier decision on JUPITER allocation `961143`. The benchmark calls
the production FlashMLA FP8 sparse-decode interface over the complete per-rank
cache footprint: 78 layers, 6,251 physical blocks, 20,470,474,752 cache bytes,
and 2,048 selected positions per layer. Each simulated token reads 104,792,064
bytes. Twenty-one distinct index sets are reused by the GLM layer groups.

The cache was tested with random, sorted, and clustered positions in both HBM
and pinned local Grace memory. Results below are 100 CUDA-graph replays after
five warmups. Host/HBM output matched exactly for every pattern, and sampled
Grace locality was 100% before and after the run.

| Pattern | HBM median | HBM p95 | Host median | Host p95 | Host penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 1.494 ms | 1.498 ms | 3.006 ms | 3.010 ms | 1.511 ms |
| Sorted | 1.483 ms | 1.486 ms | 2.967 ms | 3.043 ms | 1.484 ms |
| Clustered | 1.468 ms | 1.470 ms | 1.735 ms | 1.739 ms | 0.267 ms |

The realistic random and sorted patterns sustain about 35 GB/s from host-UVA,
versus about 70 GB/s from HBM. The v2 host-cache gate requires p95 at most
0.5 ms/token/rank, so host-UVA misses it by about 6x. Even the favorable
clustered pattern misses by about 3.5x.

With the corrected null block and 7 GB HBM reserve, plan-only places 4,228 hot
and 572 cold expert slots per rank with the main cache in Grace, versus 3,176
hot and 1,624 cold with the cache in HBM.

## Production retry

A requested full-model retry used TP4/EP4, the tiered destination loader,
`torch.compile`, full/piecewise CUDA graphs, and the complete 400K host cache.
Startup allocated 19.06 GiB of host cache per rank at 100% sampled NUMA
locality. It left 6.26 GiB HBM free before requests, above the 5.59 GiB runtime
minimum. The checkpoint load took 12:47 and engine profile, compile, cache
creation, and warmup took 150 seconds, including 99 seconds of compilation.

| Input | Cache | TTFT | TPOT | Decode | Host decode change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,096 | HBM, mean of two | 0.982 s | 27.553 ms | 36.29 tok/s | - |
| 4,096 | Host, cold seed 6 | 1.717 s | 24.717 ms | 40.46 tok/s | +11.5% |
| 399,744 | HBM | 118.385 s | 27.346 ms | 36.57 tok/s | - |
| 399,744 | Host | 262.872 s | 24.074 ms | 41.54 tok/s | +13.6% |

The host-cache server completed the exact 399,744 + 256 case and retained
2.58 GiB free HBM at the request high-water mark. A seed-5 4K result is kept
only as a prefix-cache audit: its prompt was a prefix of the preceding seed-5
400K request, so its 0.262-second TTFT is not used in the table.

The end-to-end result is more favorable to host placement than the isolated
cache kernel: freeing 20.47 GB for 1,052 additional hot expert slots cuts TPOT
by 10-12%. It also makes cold 4K TTFT 75% slower and exact-400K TTFT 122%
slower. The current v2 plan still requires host MLA p95 at or below 0.5 ms, so
AUTO remains fail-closed on HBM. Host-UVA is retained as a measured decode
alternative rather than the selected valid plan.

The next phase is routing-trace capture and static owner/residency placement
against the selected HBM budget. The trace should also explain how much of the
host-cache decode gain comes from avoiding cold-expert reads.
