# KV on Grace: what the attention kernels actually do with far memory

Status: **Complete.** Isolated kernel measurement, no server involved. KV in
Grace costs **+8.5 ms/step** at c4 on the production sparse MLA path, against a
~9.4 ms/step upside from the HBM the move would free — a wash, not a win, and
the reason is the kernel rather than the link. Two of the three numbers here
were misread on the way and are corrected below.

## The question

Decode reads the KV cache in exactly one place: the attention kernel. If that
kernel could read it over C2C from Grace LPDDR instead of HBM, the ~19.3 GiB per
rank the cache occupies would go to expert residency, which
[Phase 32](../2026-08-01-marlin-tier-overlap/README.md) named as the next lever.

Earlier attempts moved the cache with `--mla-cache-tier host` and measured a
whole server; the 2026-07-17 retry saw **+122% TTFT at 400K** and stopped. This
one tests the kernel alone, so the penalty is attributable.

## Method

Three benchmarks, all NUMA-bound explicitly — `GraceAllocation.allocate_pinned`
audits placement but *binds nothing*, and a bare `sbatch` inherits no policy, so
without `numactl --cpunodebind --membind` the allocation lands on the wrong node
at 0% locality. `detect_numa.py` resolves the paired node per GPU.

| Script | What |
| --- | --- |
| `mla_cache_full_footprint.py` | The production FlashMLA sparse fp8 kernel over a full 78-layer, 6,251-block, 400K-capacity cache |
| `dense_attn_kv_tier.py` | Dense MLA and GQA, the non-DSA kernels |
| `grace_read_ceiling.py` | Triton streaming read, scattered gather, TMA probe, copy-engine control |

Reported bandwidth is `78 layers x 2048 topk x 656 B x query_tokens / time`, so
the millisecond column is a whole-model per-step figure, not per layer.

**MTP3 sets the query-token count**: 4 at c1, 16 at c4. `tok=1` is kept as the
control against the older gate. `index-mode=shared` models MTP's consecutive
draft positions attending to nearly the same context (the realistic case);
`independent` is the pessimistic bound.

## Result 1 — the sparse MLA kernel

Job `1388432`, `index-mode=shared`, CUDA-graph replay:

| pattern | tok | HBM ms | Grace ms | penalty | HBM GB/s | Grace GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 1.482 | 3.012 | 2.03x | 70.7 | 34.8 |
| random | 4 | 1.452 | 4.182 | 2.88x | 288.6 | 100.2 |
| **random** | **16** | **2.197** | **10.691** | **4.87x** | **763.1** | **156.8** |
| sorted | 16 | 2.187 | 10.081 | 4.61x | 766.5 | 166.3 |
| clustered | 16 | 2.163 | 5.961 | 2.76x | 775.2 | 281.3 |

The c4 row is the one that matters: **+8.5 ms/step**. `index-mode=independent`
is 6% worse at tok=16 (147.4 GB/s), so MTP's index sharing is worth little.

Locality in the index set is worth a great deal, though — `clustered` reaches
281 GB/s against `random`'s 157. Real top-k sets sit somewhere between, and no
production index trace was captured, which is the main weakness of this
measurement.

**Correctness held everywhere**: all nine pattern x token combinations returned
exact-across-tiers output, so the tier is transparent to the arithmetic.

## Result 2 — the non-DSA kernels

Job `1388502`. Dense MLA never ran: the fp8 dense kernel wants
`num_blocks x num_heads_k x (block_size * 656)` rather than the sparse layout,
and after one fix attempt it was abandoned rather than chased.

GQA (Llama-3-70B geometry, 64 Q heads / 8 KV heads / 128 dim) did run:

| ctx | bs | HBM ms | Grace ms | penalty | HBM GB/s | **Grace GB/s** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1 | 0.885 | 2.559 | 2.89x | 151.7 | **52.4** |
| 32,768 | 8 | 2.838 | 20.578 | 7.25x | 378.3 | **52.2** |
| 32,768 | 32 | 11.148 | 82.244 | 7.38x | 385.3 | **52.2** |
| 131,072 | 1 | 3.417 | 10.233 | 2.99x | 157.1 | **52.5** |
| 131,072 | 8 | 11.240 | 81.985 | 7.29x | 382.1 | **52.4** |
| 131,072 | 32 | 45.103 | 329.084 | 7.30x | 380.9 | **52.2** |

**FA3 GQA sits at 52 GB/s from Grace and does not move** — not with batch, not
with context, while HBM scales 151 -> 385. A number that flat across a 32x
change in work is a structural limit, not a bandwidth one. It is **still
unexplained**; the TMA probe below rules out the obvious suspect.

## Result 3 — the ceiling probes, and two corrections

Job `1388638`. Streaming read, contiguous:

| CTAs | HBM GB/s | Grace GB/s |
| ---: | ---: | ---: |
| 132 | 662.7 | 289.2 |
| 264 | 1,286.2 | 404.1 |
| 528 | 2,341.2 | 418.6 |
| 1,056 | 3,329.8 | 407.9 |
| 2,112 | 3,653.8 | 402.8 |

| probe | Grace GB/s |
| --- | ---: |
| TMA descriptor load | **419.5** |
| copy engine H2D | 422.0 |

**C2C delivers ~400-420 GB/s, and TMA works on host-mapped UVA memory at full
speed.** Both hypotheses that would have made the link the culprit are dead.

Scattered gather, sweeping granularity (the table's `gran` column is mislabelled
in the raw output — it prints the row count; the four blocks are row sizes 128,
256, **512** and 1,024 bytes). At 512 B, nearest the 656 B MLA row:

| CTAs | HBM GB/s | Grace GB/s |
| ---: | ---: | ---: |
| 132 | 111.0 | 73.1 |
| 264 | 217.5 | 139.3 |
| 528 | 411.9 | 252.1 |
| 1,056 | 767.6 | 382.3 |
| 2,112 | 1,355.8 | 368.3 |

### The corrections

Two readings taken during this work were wrong and are recorded so they are not
repeated:

1. **52 GB/s and 157 GB/s were read as hardware limits.** They are kernel
   limits. The link does 400+. The tell was in the data already: the scattered
   gather beat the contiguous stream at some points, which no memory system can
   do.
2. **The gap between 157 and 361 GB/s was read as 2.3x of headroom.** It is not.
   The 361 figure came from a gather running 1,056 CTAs; the MLA kernel runs
   **132**, because 224 KiB of shared memory admits one CTA per SM. At matched
   parallelism a plain gather gets **73 GB/s** and the MLA kernel gets 157 — it
   is already 2.2x *better*, not 2.3x worse.

The second correction is what [`2026-08-06-mla-grace-kernel`](../2026-08-06-mla-grace-kernel/README.md)
was built to act on before it was found, and it is why that experiment stopped
at its first gate.

## Verdict

| | |
| --- | ---: |
| c4 attention penalty, KV on Grace | +8.5 ms/step |
| HBM freed | ~19.3 GiB/rank |
| Estimated residency gain from that HBM | ~9.4 ms/step |

A wash on decode, before counting prefill or the fact that the cold expert tier
already contends for the same link (Phase 26 measured it at 379 GB/s against a
373 GB/s achievable rate). Not shipped. The follow-on kernel work is in
[`2026-08-06-mla-grace-kernel`](../2026-08-06-mla-grace-kernel/README.md).

## Files

| File | What |
| --- | --- |
| `job.sh` | Sparse MLA footprint sweep, both index modes |
| `job-dense.sh` | Dense MLA + GQA |
| `job-ceiling.sh` | Streaming, gather, TMA, copy-engine probes |
| `footprint-{shared,independent}-*.txt` | Sparse results + JSON |
| `dense-*.txt` | GQA results (dense MLA rows failed) |
| `ceiling-*.txt` | Ceiling probes; `1388638` is the complete one |
