# Anchor-layer KV off HBM: how much would it cost? (2026-10-19)

The user: can the 22 anchor layers' main MLA KV (the layers with their own DSA
indexer; HBM under `skip_host_uva`) move to Grace, even partly, at ~zero
decode cost? Per rank at the 800K pool that is 22 x 656 B x 200K = 2.69 GiB
(~530 hot experts); the indexer cache (0.54 GiB) has to stay (it is scanned
in full every step). Per 100K pool tokens HBM would drop from 0.404 to 0.068
GiB per rank.

## Prior art: SGLang HiSparse (local sglang-upstream, 2026-08-29)

Full KV for every layer in pinned host memory; per request and layer a GPU
hot buffer (default 2 x top-k = 4096 slots, configs 4096-6144) with LRU, plus
a reserved slot for the newest token. Each layer: indexer top-k -> swap-in
kernel (hash lookup per request, misses evict LRU and copy host->device) ->
FlashMLA on device rows only; synchronous on the anchor's critical path.
IndexShare skip layers replay the anchor's miss plan on a side stream (our
stager's idea). Requires `--disable-radix-cache`; MTP support is in open PRs
(#40670, #41780/1). Their numbers: TPOT +17-18% static (#30266, 16K in),
swap-in 4.16 -> 2.04 ms of a ~23 ms GB300 step (#43019), lookup alone 7-25
us per layer warm (#31341, H200). Their small-batch fix is "dynamic
residency" (#23935, #30266): keep full KV on GPU until the pool is under
pressure. No published hit rates.

## Slack before an anchor's attention (prod traces, mg2a, device 0)

`slack.py`-style analysis (inline in this session) on the run-2 prod arm
traces: for every anchor FlashMLA call, the time from the end of the DCP
top-k merge kernel (`dsadcp_indexer`, the last kernel producing the final
top-k) to the FlashMLA start, and what runs between.

| trace | anchors | window (med) | top-k-independent kernels in it | convert + sched |
|---|--:|--:|--:|--:|
| dec1 ~5K (M=4) | 2904 | 18.8 us | 10.8 us | 5.3 us |
| dec1 ~50K (M=4) | 2712 | 18.8 us | 10.8 us | 5.3 us |
| dec4 ~5K (M=16) | 482 | 20.3 us | 14.6 us | 5.1 us |
| dec4 ~50K (M=16) | 1944 | 20.6 us | 14.8 us | 5.2 us |

The independent work is the layer's KV write, the W_UK absorb GEMM and the
q concat. So an anchor could stage its own rows from Grace on a side stream
forked right after the top-k merge, overlapped with those ~11-15 us.

Today's stager kernels (medians, per group = anchor's indices, 3 skip layers):

| trace | convert | mark | compact | remap | gather (3 layers) |
|---|--:|--:|--:|--:|--:|
| dec1 ~5K | 4.0 | 2.1 | 5.5 | 1.8 | 15.3 |
| dec1 ~50K | 4.0 | 2.5 | 5.5 | 3.4 | 16.1 |
| dec4 ~5K | 3.7 | 3.2 | 5.1 | 4.6 | 33.7 |
| dec4 ~50K | 3.8 | 3.6 | 4.9 | 4.4 | 59.3 |

Self-staging chain with these kernels: M=4 ~15 us bookkeeping + ~5 us gather
vs ~13 us of main-stream work -> FlashMLA ~3 us later per anchor (~0.07
ms/step); M=16 at 50K ~17 + ~20 us -> ~16 us later (~0.35 ms/step). The
gather at M=16 is near C2C bandwidth (~5 MB per layer in ~20 us), so only
fewer rows (a hot buffer's hits) shrinks it; the bookkeeping can shrink
(compact scans every slot; mark/compact/remap could be one kernel, shared
with the skip layers' copy).

## Prefill under DCP (skip layers already pay a Grace read)

Prefill does not read Grace from FlashMLA: every layer (anchor or skip) runs
`cp_gather_and_upconvert_fp8_kv_cache` over its whole local shard into an HBM
bf16 workspace (`dcp_sparse_prefill`, flashmla_sparse.py:289-324),
synchronously, no prefetch. At a 56K-token shard: anchors 30 us, skips 93 us
(~400 GB/s from Grace), so the skip penalty is 3.6 ms per 611 ms chunk
(0.6%). Re-upconverting the full shard every layer costs ~1% of a chunk
whether the rows come from HBM or Grace; anchor KV on Grace would add the
same ~63 us per anchor layer (~1.3 ms per chunk, 0.2%).

## Top-k locality: LRU hot buffer on real c=4 agentic traffic

`topk-dump-pf10.diff` on worktree pf10 (= prod 44153148ba):
`VLLM_ANCHOR_TOPK_DUMP=<dir>` records on rank 0 every decode step's 21
target anchors' converted top-k (rank-local slots, width 768, ~519 valid per
token), new-token slots and token -> batch row; `launch_dump.sh` runs the c=4
agentic timeline (900 s, contexts ~290-330K global) on two holds (akv-d1,
akv-d2: 17.2K / 17.8K decode steps, 73% / 69% at M=16). (First attempt hung:
`written[i] = 1` is a CPU->GPU copy, illegal in capture; now `fill_`.)
`sim_lru_nb.py` (numba) replays them; the step's new tokens are inserted by
the KV write and never miss. `req` = one LRU per (request, layer), dropped
when the request leaves the batch; `global` = one LRU per layer with 4x the
capacity (same total rows). Warm = steps where every request is >= 8 steps
old (barely differs: new turns reuse rows already resident).

Misses per anchor call (summed over the batch) on akv-d1, warm; akv-d2 agrees
to within 0.2 pp:

| M | rows read | policy | cap 1024 | 2048 | 4096 | 8192 |
|---|--:|---|--:|--:|--:|--:|
| 16 | 3558 | req miss% | 41.7 | 15.3 | 7.4 | 4.0 |
| 16 | | req mean / p99 rows | 1482 / 4209 | 543 / 1752 | 265 / 946 | 141 / 517 |
| 16 | | global miss% | 35.6 | 13.8 | 6.3 | 2.7 |
| 16 | | global mean / p99 | 1266 / 4642 | 492 / 1584 | 223 / 783 | 95 / 371 |
| 4 | 839 | req miss% | 31.4 | 11.1 | 4.8 | 2.3 |
| 4 | | global miss% | 6.0 | 3.6 | 2.2 | 1.3 |

A request reads ~890 distinct rows per layer per step (its 4 query tokens
overlap), so caps below ~1024 thrash. HBM per rank for the buffers (4
requests x 21 layers x 656 B): cap 2048 108 MiB, 4096 215 MiB, 8192 430 MiB,
against 2.69 GiB for the full anchor KV at the 800K pool: 2.48 GiB (~500 hot
experts) saved at cap 4096.

Cost on the anchor's critical path (top-k is known only after the merge): at
cap 4096, M=16 fetches a mean 223-265 rows (146-174 KB) per anchor, p99
~800-950 rows (~0.6 MB); at C2C rates that is 1-3 us of transfer, so the
cost is the lookup/insert bookkeeping, not bandwidth. Today's stager chain
(mark + compact + remap ~12-15 us) would not fit the 11-15 us window; a
fused lookup+gather kernel would need to run in ~10 us to keep the anchor's
FlashMLA on time, else ~21 x few us = ~0.1 ms/step (~0.4%). M=4 (one
request) misses ~40-50 rows per anchor at cap 4096 (global: ~15-20).

## Hot-cache kernel (kernel/, 2026-10-19)

Policy sim (`sim_policy_nb.py`, one buffer per layer shared by the batch,
16384 rows, M=16): exact LRU 6.5-6.6% miss (~228 rows/call), CLOCK (ref bit,
skip rows read this step) the same, FIFO ~1-1.5 pp worse; inserting the
step's new tokens into real rows instead of reserved ones changes nothing.
Per layer the claims differ a lot (layer 0 ~22/step, 1 ~300, 10 ~235, 20
~275): per-layer capacity is a later lever.

`anchor_hot.cu` maps one anchor's converted top-k to hot-buffer rows
(N=16384 + 64 reserved rows for this step's tokens) and copies the missed
rows Grace -> HBM; entries keep position, multiplicity and -1, so FlashMLA
reads the same bytes in the same order. `replay.py` replays the akv-d1 dump
(rank 0) for one layer and checks every entry byte for byte against the
Grace source, the overflow counter and the where/slot_of inverse; `--prof`
builds in-kernel globaltimer splits. All versions: 0 bad entries on login
(layers 1 eager, 10 and 20 graph) and Booster.

Booster (GPU 0, NUMA 0, C2C 421 GB/s), CUDA graph, M=16 span median
(5000 profiled steps):

| version | launches | layer 1 | layer 10 |
|---|--:|--:|--:|
| v1: lookup, select (per-warp CLOCK, atomic hand), copy, finish | 4 | 15.46 us | 16.42 |
| v2: select in lookup's last block (block-scan sweep), finish in copy's last block | 2 | 15.65 | 14.91 |
| v2b: finish in lookup's last block, parity counters; copy has no barrier | 2 | 14.21 | 14.01 |
| v3: one kernel; last block sweeps, others spin on a release flag, then copy | 1 | 14.30 | 13.50 |
| v3b: plain __ldcg loads in the sweep | 1 | 14.05 | 13.57 |
| v3c: no where[old] check (4096 random loads from one SM, ~2 us) | 1 | 11.49 | 11.07 |
| v3d: copiers read Grace while the sweep runs; sweeper not a copier | 1 | 10.30 (p99 12.26) | 10.02 (p99 12.23) |

v3d split (layer 1 / 10): lookup body 3.9 / 3.6 us, sweep 4.2 / 3.9 (candidate
loads 1.2, block scan 0.4, stores 1.8 / 1.5), post-release ~2.1. Lessons:
the cost is serial latency in one block and one SM's request rate, not
bandwidth (300 rows x 656 B is ~0.5 us of C2C); relaxed atomics vs plain
loads made no difference. Budget (see "Slack" above): the side-stream chain
is conversion (~4-5 us) + map; it delays FlashMLA only past ~19-20 us, and
is fully hidden behind the 11-15 us of top-k-independent work if the map is
<= ~6-10 us, before contention with that work.

Astra review v1: `kernel/astra-v1-review.md`, disposition in
`kernel/astra-v1-disposition.md` (conditional eviction adopted in v2, then
replaced in v3c by the inverse invariant + audit). HBM correction: 21 anchors
are 2.57 GiB/rank at the 800K pool (2.69 was 22 layers); after the buffers and
tables ~2.33 GiB/rank freed.

## vLLM integration (worktrees/ahc-b1 = e9d65055e3, branch anchor-hot-cache)

`VLLM_ANCHOR_HOT_CACHE=same|free` (under skip_host_uva) moves the 21 target
anchors' main cache to Grace; the MTP layer stays in HBM. Each anchor gets
`anchor_hot_cache.AnchorHotCache`: 16384 + 64 reserved rows, built from
anchor_hot.cu (v3e, new_slots now int64 = slot_mapping) under VLLM_CACHE_ROOT.
- Decode step: the KV write also puts this step's tokens into the reserved
  rows. `start_mqa` (MLAAttention.forward_impl, before the q absorb and
  its DCP all-gather) forks a side stream that runs the DCP conversion and
  then the map. forward_mqa waits for it and runs FlashMLA on the hot buffer.
  The stager still gets the Grace cache and the original slots.
- Any other step reads Grace directly and drops the written slots
  (`_invalidate_kernel` in do_kv_cache_update).
- Planner: host += 21 anchors; "same" keeps the HBM charge, so the hot set
  matches prod (14,284 hot per rank); "free" charges only the hot tables.
- `VLLM_ANCHOR_HOT_CACHE_CHECK=1` reruns each mapped FlashMLA call on Grace
  and counts mismatching bits.

In-situ check (`launch_check.sh`, hold 2412782, dec4): ~11.4K mapped calls
per rank, 0 mismatches. Free HBM after startup is 5.14 GiB (prod
2.84–2.91 GiB). Astra review: `kernel/astra-int-*.md` (no blocker; ship
items listed). ABC campaign: `launch_abc.sh`, compared by `ak_compare.py`.

## Hot-cache size from the decode step (ahc-dev, after the c=4 ABC tree)

rows = 4/3 of the entries a decode step can map (max_num_seqs x (1 + k),
capped at 64, x the 768 DCP width), rounded to 64, at least 4096:
c=1 MTP7 8192, c=4 MTP3 16384 (unchanged), c=8 32768, c=16 MTP3 65536.
No decode step falls back to Grace; only prefill reads it. The pool size
only scales the slot->row table (4 B per slot per layer). Adversarial,
graph and stress tests at 8192 / 65536 / 16384 rows (`adversarial.py
--rows`): 0 failing steps, overflow 0, no hang
(`logs/ahc-rows-adversarial-login.log`). Kernel time at Q=64 is not
measured yet (needs a c=16 dump).

## Served ABC (2026-10-19, `launch_abc.sh`, 6 holds, c=4 launcher config)

a = prod (ahc-a = 24175ed0d3), b = ahc-b1 `same` (14,284 hot experts, as a),
c = ahc-b1 `free` (14,773 hot, 4,427 cold, +489 hot experts per rank).
`ak_compare.py` (`ak_compare.txt`): verify steps M=16, medians per arm,
x - a per hold:

| kind | holds | span b-a | rest b-a | span c-a |
|---|--:|--:|--:|--:|
| dec4 ~5K | 2 | -0.296 ms | -0.189 ms | |
| dec4 ~50K | 2 | -0.143 ms | -0.171 ms | |
| tlnsys (real c=4 replay) | 4 (b), 2 (c) | -0.042 ms | -0.188 ms | -0.569 ms |

(The step is ~22.5 ms; rest = span minus the MoE layer kernels.) The map
kernels, 21 per step at 10–12.7 µs each (208–267 µs of kernel time), run on
the side stream and do not lengthen the step. FlashMLA time is unchanged
(`ak_kdiff.py`). Rest is 0.17–0.19 ms lower in b in all 6 holds, a cause
not attributed yet (nothing on the anchor path got faster). Matched-request
TTFT (`ak_ttft.py`): big prefills b/a 0.998–1.004 (anchors read from Grace
in prefill cost nothing visible); small TTFT -4 to +27 ms (queueing noise).
c's extra 489 hot experts take 0.57 ms off the step (~2.5%).
Free HBM after startup: a 2.88, b 5.14, c 2.83 GiB.
