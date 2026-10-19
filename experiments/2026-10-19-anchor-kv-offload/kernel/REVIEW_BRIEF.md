# Anchor hot cache: review brief

## Context

GLM-5.3 (DSA sparse attention, MLA, fp8_ds_mla KV 656 B/token/layer) served
on 4x GH200 with vLLM, TP4 + DCP4 (KV tokens sharded over 4 ranks), MTP3
speculative decode (4 query tokens per request per step), c=4 requests,
CUDA graphs (FULL graphs for decode). Tiered MoE: experts that don't fit in
HBM live on Grace; every GiB of HBM freed = ~200 more hot experts.

KV tier today (`mla_cache_tier=skip_host_uva`): the 21 target "anchor"
layers (own DSA indexer) + MTP layer 78 keep their main KV in HBM; the 57
"skip" layers (reuse the previous anchor's top-k) keep theirs in pinned Grace
memory (UVA). After each anchor's FlashMLA, a side stream (`SkipKVStager`,
skip_kv_stage.py) copies the rows the anchor's top-k selects for its 3 skip
layers into HBM staging buffers and remaps the indices; skip layers write
this step's tokens into reserved rows. Anchor main KV at the 800K pool:
2.69 GiB per rank.

Goal: move the 21 anchors' main KV to Grace too, with an HBM hot buffer per
anchor, at ~zero decode cost. Same ops only: FlashMLA must read the same
bytes in the same order (indices keep position, multiplicity and -1), so the
attention output is bitwise identical.

## Evidence so far

- Top-k dumps from real c=4 agentic traffic (17K decode steps x 2 runs,
  contexts ~290-330K global, 73% of steps M=16 tokens): per anchor and step
  the batch reads ~3558 distinct rows per rank (M=16), ~839 (M=4); each
  token selects ~519 valid of 768 columns (DCP-converted, rank-local).
- LRU sim: one buffer per layer shared by the batch, 16384 rows: 6.2% miss
  (~220 rows/anchor call, p99 ~780); per-request buffers 4096 rows each:
  7.4%. CLOCK (ref bit, skip rows read this step) matches exact LRU; FIFO
  is ~1-1.5 pp worse. Inserting this step's new tokens into real rows (vs
  reserved rows) changes nothing.
- HBM: 21 x 16384 x 656 B = 215 MiB vs 2.69 GiB: ~2.48 GiB/rank freed.
- Critical-path window: from the end of the DCP top-k merge kernel to the
  anchor's FlashMLA start there is 18.8 us (M=4) / 20.5 us (M=16), of which
  10.8 / 14.8 us is top-k-independent work (KV write, W_UK absorb GEMM, q
  concat) and ~5 us the index conversion + schedule. A side stream forked at
  the merge could hide ~11-15 us. FlashMLA itself is ~10-13 us.
- Today's stager kernels for comparison (mark 2-3.6, compact 5, remap 2-4.6,
  gather of 3 layers 15-59 us).

## Proposed design (v1)

Per anchor layer, on HBM: buffer H (N=16384 rows + 64 reserved rows, laid out
as the FlashMLA cache, (rows/64, 64, 656)), `where[S]` (slot -> row; S ~ 200K
slots per rank), `slot_of[N]`, `stamp[N]` (step last read), `ref[N]` (CLOCK
bit), counters. Global per layer (not per request): prefix-shared blocks stay
cached across turns.

Per anchor per decode step, after the DCP index conversion and before
FlashMLA (anchor_hot.cu, 4 launches):
1. lookup: invalidate this step's new-token slots (KV write changes them);
   entries: -1 -> -1, new token j -> N + j (reserved row, filled by a second
   KV write into H, as the skip layers do), hit -> row (stamp, ref=1), miss ->
   pending list; first miss per slot claims where[s] -1 -> -2 (atomicCAS) and
   is listed. Warp-aggregated appends.
2. select: per warp 32 claims; CLOCK candidates from one atomic per round on
   the hand; skip rows stamped this step; second chance on ref; victim owned
   via atomicExch(stamp, step); evict (where[old] = -1), publish where[s].
3. copy: Grace -> H for every missed row, one 16 B word per thread (41 per
   row), 4 loads before 4 stores; then pending entries read where[s].
4. finish: step++, counters reset.

Planned integration:
- New tier (anchors' main caches on Grace; layer 78 stays in HBM).
- KV write: normal write goes to Grace (UVA); a second write puts this
  step's tokens into H's reserved rows (top-k independent, in the window).
- Invalidation outside decode: prefill / any KV write to slot s must drop
  where[s] in all 21 layers (blocks freed and reallocated to another request
  would otherwise be served stale). Plan: a scatter kernel over the step's
  slot_mapping on every step that writes KV (prefill and decode).
- Prefill reads the anchor cache from Grace through the existing full-shard
  upconvert gather (costs ~63 us per anchor per 56K-token chunk, ~0.2%).
- Placement: fork lookup/select/copy on a side stream right after the DCP
  top-k merge (needs the index conversion moved there too), join before
  FlashMLA.
- Skip layers unchanged (they stage from their own Grace caches using the
  anchor's converted indices, which stay slot-valued).

## Questions for review

1. Correctness of anchor_hot.cu: races between lookup/select/copy, the claim
   protocol, victim ownership (hand wrap within a step), eviction of a slot
   that another entry hit this step, the -2 state on overflow, memory
   ordering between kernels, int overflow of the hand.
2. Design holes: speculative decode (rejected draft slots rewritten next
   step), block free/realloc, prefix-cache hits on blocks written by another
   request, DCP (slot_mapping is -1 for tokens other ranks own), padded FULL
   graph batches, CUDA graph capture of the counters, the MTP drafter reading
   anchor layers?, prefill/decode mixing.
3. Cost: is 4 launches + atomics realistic inside ~10-15 us at ~220 (p99
   ~800, max ~3500) misses per call? What would you fuse or restructure
   (cooperative single kernel, persistent kernel, PDL)? Better policy than
   CLOCK for a GPU?
4. Anything simpler that reaches the same HBM saving.
