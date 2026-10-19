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

## Top-k locality (pending)

`topk-dump-pf10.diff` on worktree pf10 (= prod 44153148ba):
`VLLM_ANCHOR_TOPK_DUMP=<dir>` records on rank 0 every decode step's 21
target anchors' converted top-k (rank-local slots), new-token slots and
token -> batch row; `launch_dump.sh` runs the c=4 agentic timeline (900 s)
on two holds; `sim_lru.py` replays it through per-request and global LRU
buffers.
