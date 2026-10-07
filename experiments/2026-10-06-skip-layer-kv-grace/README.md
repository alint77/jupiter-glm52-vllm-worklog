# Skip-layer MLA KV on Grace, prefetched from the anchor's top-k (plan v2, 2026-10-06)

Status: **implemented (vllm ad589f557c, a6043fad30), awaiting a node** (Booster maintenance, 2026-10-06). v2 folds in the Codex review of v1
(`codex-review-v1.md`, gpt-6-astra xhigh); the last section maps each finding
to its change here.

Goal: move the main MLA KV of GLM-5.3's 57 index-share ("skip") layers to
Grace and copy each step's selected rows into HBM before those layers'
attention runs, so attention still reads HBM and the step pays (nearly)
nothing. The freed HBM (~3.4 GiB per GPU) goes to hot experts. Anchor layers
(own indexer) keep their KV in HBM. The "KV stays on GPUs" scope rule does not
apply to offload with no step overhead (memory `glm-serving-scope`).

## Why this can be free when KV-on-Grace was not

- 2026-08-05-kv-grace-attn / 2026-08-06-mla-grace-kernel: FlashMLA sparse
  reading KV **directly** from Grace gets 157 GB/s (occupancy-bound): +8.5
  ms/step at c4. Attention must keep reading HBM.
- The same work measured plain gathers of 656 B rows from Grace at 361-384
  GB/s, TMA at 420 GB/s: moving the rows is cheap if it happens early.
- GLM-5.3 config `index_topk_freq=4, index_skip_topk_offset=3`
  (deepseek_v2.py:1098-1112): anchors are layers {0, 1, 2, 6, 10, ..., 74}
  (21); anchor 4k+2 is followed by skip layers 4k+3..4k+5 (19 groups, 57
  layers) that reuse its top-k (confirmed by the review). flashmla_sparse
  converts an anchor's indices to this rank's physical slots and the skip
  layers reuse that result (`_dcp_converted_indices`, `_reuses_topk`,
  flashmla_sparse.py:917-967). Under full-graph capture this is a capture-time
  reuse: replay reruns the recorded conversion and consumers, never the
  Python cache, so it stays correct per replay (review point 4).

## Budget (per GPU, 400K, DCP4, planner log of job 2183109)

| | today | plan |
|---|--:|--:|
| main MLA KV, 21 anchor layers (HBM) | 1.28 GiB | 1.28 GiB |
| main MLA KV, 57 skip layers | 3.49 GiB HBM | 3.49 GiB Grace |
| HBM staging: 3 buffers x R rows x 656 B, R = 16384 + 64 | 0 | 30.9 MiB |
| **freed HBM** | | **~3.45 GiB = ~174 hot experts** |

Block capacity must not change: the planner's available-memory figure is
cross-tier capacity accounting that derives the block count
(tiered_moe_kv.py:141-235, kv_cache_utils.py:1395), so the skip bytes move
to the host side of that account with num_blocks unchanged; only the HBM side
shrinks (review point 9).

Expected gain, rough: cold experts per GPU per layer ~2.00 -> ~1.88 over the
75 MoE layers at ~51 us each: ~0.46 ms/step (~1.9%). Gate 0b replaces this
with a replay, including the slowest rank per layer (the all-reduce waits for
it).

Copy traffic: 57 x U x 656 B per step, U = this rank's union of selected rows
over the verify queries.

| U (per rank, per group) | per step | link time at 361-420 GB/s |
|--:|--:|--:|
| 1,000 | 37 MB | 0.09-0.10 ms |
| 16,384 (worst case) | 613 MB | 1.46-1.70 ms |

So "free" holds only for the U distribution real traffic produces; gate 0a
measures it and sets the go/no-go (review point 8).

## Design

### Buffers

One shared set of 3 HBM buffers, one per position in a group, each laid out
like a KV cache of (R/64) blocks x 64 x 656 B:
- rows [0, 16384): **gathered rows** (worst case: 8 queries x 2048 distinct
  local rows);
- rows [16384, 16448): **reserved rows**, a fixed region: verify position j
  -> row 16384 + j (8 used, padded to one block).

Gathered and reserved regions are disjoint, so the reserved-row write and the
gather never touch the same rows, and R covers the worst case with no
overflow path (review point 1).

Single buffering is enough: main-stream order is skip(g-1) x 3 -> anchor g ->
skip(g) x 3, and group g's gather forks after anchor g's conversion, so group
g-1's readers are done (confirmed by the review, including across steps).

### Per step, per group g (decode, full CUDA graph)

1. **Anchor g, right after `_dcp_converted_indices`** (inside its attention,
   just before its FlashMLA call, flashmla_sparse.py:859): record an event and
   fork a side stream:
   - **plan kernel**. Inputs: the converted indices [T, 2048] (this rank's
     physical slots, compacted prefix, -1 tail), and this step's KV-write slot
     mapping for the verify tokens. Output: the union of selected slots that
     are *not* this step's slots, assigned rows 0..U-1 in slot order (a
     bitmap over the rank's whole physical slot space, num_blocks x 64, then
     a prefix sum; deterministic). Plus remapped indices [T, 2048] built
     **elementwise**: each entry keeps its position, multiplicity and -1;
     historical slots -> their gathered row, this step's slot of verify
     position j -> reserved row 16384 + j.
   - **gather kernel**: copy the U rows from each of the 3 skip layers' Grace
     stores (UVA aliases) into the 3 buffers' gathered region. Record e_g.
2. **Skip layer** (position p in group g):
   - KV write: `FlashMLASparseImpl.do_kv_cache_update` (override of
     backend.py:1025) writes the usual Grace slots and also buffer p's
     reserved rows with the static mapping j -> 16384 + j. Both call paths
     reach the impl (direct call, mla_attention.py:611-620; custom op
     `unified_mla_kv_cache_update`, mla_attention.py:1110-1116), so the hook
     belongs in the impl, not at a call site (review point 3). The write
     depends only on this layer's own inputs; the planner never writes
     reserved rows (review point 2).
   - Wait e_g, then the unchanged FlashMLA sparse decode on (buffer p,
     remapped indices, same valid counts and LSE masking as today).

This step's tokens are selectable: the indexer inserts their keys before
top-k (sparse_attn_indexer.py:380, 538), so a later verify query can select an
earlier one. Their skip-layer KV only exists once that layer runs, so it can
never be prefetched: excluding this step's slots from the gather means stale
bytes left in Grace by an earlier rejected verify are never read (review
point 2).

Lead time: from the anchor's conversion to the first skip layer's attention
is the rest of the anchor layer (its FlashMLA, DCP combine, o_proj, all-reduce,
MoE, ~200 us), not a full layer (review point 8). The copy at U = 1,000 is
~5 us per group.

### Other paths (correct, not optimized; prefill is not a priority)

- **Eager prefill (dcp_sparse_prefill)**: upconverts the rank's whole local
  shard, which now streams from Grace (flashmla_sparse.py:278). Its cost on
  Grace is unmeasured; gate 4 reports TTFT.
- **Captured prefill pieces (<= 1024 tokens)**: the `dcp_sparse_prefill`
  branch is skipped while capturing (flashmla_sparse.py:849), so these
  record the mixed fp8 path, which reads physical slots straight from the
  cache, i.e. from Grace at FlashMLA's ~157 GB/s. Correct but possibly slow;
  if gate 4 shows a TTFT regression, stage the chunk's union like decode
  (review point 6). The decode staging must never key off
  `_dcp_converted_indices(..., prefill=True)`, whose indices address the bf16
  workspace, not Grace slots.
- Eager decode: read the Grace store directly.

### Allocation

- `config/tiered_moe.py` `MLACacheTier`: new `"skip_host_uva"`.
- `kv_cache_utils.py:1405` / `tiered_moe_kv.get_tiered_kv_memory_tier`: tier
  per KV tensor by layer (skip -> host_uva, anchor -> hbm); today it is per
  spec kind.
- `v1/worker/gpu/attn_utils.py:_allocate_kv_cache` (V2 runner, prod): port V1's
  host_uva branch (gpu_model_runner.py:7224-7262) including its packed-layout
  rejection and keeping the `GraceAllocation` owners alive for the cache's
  lifetime (V2 returns raw tensors today, attn_utils.py:183), NUMA-bound to the
  GPU's Grace node (review point 9).
- `TieredKVCachePlan`: skip bytes on the host side, buffers on the HBM side,
  num_blocks unchanged; the planner hands the HBM difference to hot experts.

## Exactness: what is and is not claimed

- **Kernel level (gate 1): bit-exact.** For the same converted indices,
  FlashMLA on (buffer, remapped) equals FlashMLA on (full cache, original),
  since every entry addresses the same 656 bytes in the same position.
- **End to end: not automatically bit-exact.** The DCP conversion compacts
  with an atomic allocator and leaves prefix order unspecified
  (sparse_utils.py:304-310), so even baseline vs baseline may differ in
  summation order. Gate 3 first measures baseline-vs-baseline determinism; if
  the baseline itself is not bit-stable, the end-to-end check is acceptance
  and GSM8K parity, not token identity (review point 7).
- `extra_k_cache` is dropped: it adds tokens to every query instead of
  substituting addresses, so it cannot express per-query selection (review
  point 7).

## Gates

**0. Measure before building (one capture run, the rest offline).**
- 0a. Index capture: env-gated in-graph copy of each anchor's converted
  indices plus the step's slot mapping into a device ring, D2H between steps,
  over **thousands** of decode steps of agentic task-set traffic (p99.9 needs
  them). Per group and rank: U distribution, how often this step's tokens are
  selected, step-to-step churn.
- 0b. Residency replay over the live routing capture
  (routes-datasets/glm53-cc-20261004-job2173771) with ~174 more hot experts per
  GPU: mean and **slowest-rank** cold experts per layer -> expected ms/step.
- **Go** if 0b's gain at the slowest rank >= 0.25 ms/step and 0a's U at p99.9
  keeps the per-group copy well inside the ~200 us lead (U <= ~8,000: ~40 us
  for 3 layers) with a mean link cost <= 0.15 ms/step.

**1. Kernels and unit tests (no server).** Plan + gather kernels. Bit-exact
FlashMLA comparison over: -1 tails, DCP compaction, ranks with no selected
rows (valid count 0, LSE masking), every acceptance length 1..8 and padded
verifies, this step's slots selected by later queries, stale bytes in this
step's Grace slots (must not be read), worst-case U = 16,384 + reserved rows,
block-boundary crossings, arbitrary block ids (recycled / prefix-hit blocks).
Gather bench from a NUMA-bound Grace store: target >= 350 GB/s at 656 B.

**2. Graph replay test.** Capture the decode graph once, replay it many times
with changing metadata contents at the same addresses (block tables, slot
mappings, lengths, indices); compare each replay to an HBM-resident reference.
Covers the side-stream fork/join inside capture and the reuse chain
conversion -> plan/gather -> skip attention (review points 4, 5).

**3. Allocation + exactness.** The new tier on V2; planner log shows skip bytes
in Grace, num_blocks unchanged at 400K, the extra hot experts; startup and the
memory-peak check pass; prefix caching reuses blocks across turns. Determinism
baseline first, then the exactness check above. GSM8K 400.

**4. A/B on the node.** Interleaved same-node runs of prod vs skip_host_uva on
the agentic task set: step time (`VLLM_STEP_TRACE_FILE`), acceptance, TTFT. A
profiler trace of each, read for **exposed** delay: skip attention starting
later than its predecessor's end because of e_g, and cold-expert C2C streams
lengthened by overlapping gathers. Also a skip_host_uva run with the
**original** hot set, to separate the copy overhead from the residency gain
(the bigger hot set can otherwise mask it) (review point 8).

## Risks

- U's real distribution (gate 0a): with little overlap across the 8 verify
  queries, the copies stop being free.
- Gathers overlapping a cold-bound MoE window take link time from cold experts
  byte for byte; gate 4's trace and the original-hot-set run show it.
- The FlashMLA metadata (`cache_seqlens`, tile scheduler, dummy block table)
  must accept a k_cache of R/64 blocks; gate 1.
- Captured small prefills reading Grace at ~157 GB/s: possible TTFT regression
  at 512-1024 tokens; gate 4.

## Review findings -> changes (codex-review-v1.md)

| # | finding | v2 |
|---|---|---|
| 1 | R = 16384 leaves no room for reserved rows | R = 16384 + 64, disjoint regions |
| 2 | reserved-row write ordering; stale rejected bytes | static j -> reserved row mapping, planner never writes it; this step's slots excluded from the gather |
| 3 | KV-write hook only on the direct-call path | override in the impl's `do_kv_cache_update`, reached by both paths |
| 4 | "once per anchor" under capture | stated as capture-time reuse; graph replay test (gate 2) |
| 5 | lifecycle coverage (acceptance lengths, recycled blocks, empty ranks, LSE) | added to gates 1-2; bitmap spans the physical slot space |
| 6 | captured prefill takes the mixed fp8 path | documented; never key staging off prefill conversion; TTFT gate |
| 7 | bit-exactness overstated; extra_k_cache additive | kernel-level claim only; determinism baseline first; extra_k_cache dropped |
| 8 | copy cost / lead time optimistic | worst case 1.5-1.7 ms stated; lead = rest of the anchor layer; go/no-go on U; original-hot-set run |
| 9 | allocation lifetime, capacity accounting | keep GraceAllocation owners; num_blocks unchanged |

## Progress (2026-10-06)

**Gate 0b passed** (`gate0b_replay.py`, 28,698 live verify steps, 75 MoE layers,
~51 us per cold expert):

| hot set per GPU | cold / GPU / layer | slowest GPU | saving vs served |
|---|--:|--:|--:|
| served, 3,180 | 1.996 | 3.425 | - |
| +174, the planner's promotion (expert-id order) | 1.743 | 3.073 | 0.97 ms/step mean, 1.35 slowest |
| +174, frequency-ranked (a rebuilt profile) | 1.676 | 2.964 | 1.22 mean, 1.76 slowest |

Far above the 0.25 ms bar and above the plan's 0.46 ms estimate. The planner
fills extra slots in expert-id order (`runtime_hot` mirrors it), so a profile
rebuilt at the larger budget is worth another ~0.25-0.4 ms.

**Implementation** (vllm `ad589f557c`, local): `mla_cache_tier=skip_host_uva`;
`v1/attention/backends/mla/skip_kv_stage.py` (plan = sort-based dedupe with
fixed shapes, graph-capturable; Triton row gather bounded by the device-side
count; reserved-row write); hooks in `flashmla_sparse.py` (stage after the
anchor's conversion, swap in the buffer for skip layers, impl-level
`do_kv_cache_update`); planner split (anchors' main cache HBM, skip layers'
host, 31 MiB staging HBM); V2 host-UVA allocation. `a6043fad30` leaves the
tier out of the compile hash (no recompile when switching).

**Gate 1 (login-node GPU, correctness only):** `tests/v1/attention/
test_flashmla_sparse_dcp.py::test_skip_layer_staging_is_bit_identical`
[1, 8 queries]: FlashMLA output and LSE on (staged buffer, remapped rows)
equal those on (full cache, original indices) bit for bit, with repeats, -1
tails, and this step's tokens selected by later queries. A mutation (not
excluding this step's slots) makes both cases fail, so the test sees stale
bytes. The suite's other 5 tests still pass.

**Next, on the node** (`run_ab.sh`, hold 2196637 queued): skip arm then prod
arm on one node, 16 agentic requests each, greedy check, one profiler window,
the staged-row histogram (gate 0a) via `VLLM_SKIP_KV_STATS`.

## First node runs (2026-10-06 evening, holds 2196637 / 2197782 / 2198220 / 2198399)

Driver `run_ab.sh` (agentic task set via `../2026-09-28-agentic-decode-bench/
bench_node.sh`, 16 requests per arm, greedy check, one profiler window);
trace analysis `skipkv_trace.py` (steps segmented by their 78 FlashMLA calls).

- **Planner**: skip_host_uva serves **3,358-3,360 hot experts per GPU vs
  3,180** (+178), num_blocks unchanged (400,255 tokens).
- **First start failed** at full decode-graph capture ("capturing stream has
  unjoined work"): with VLLM_SKIP_KV_STATS the histogram update ran on the side
  stream after the ready event (vllm a7cfa35412; graph replay test added).
- **Gate 0a** (staged rows per GPU per group, 4,940 group-steps, ctx 3-9K):
  mean ~615, p90 <= 1,024, max <= 1,280 of the 16,384 provisioned: ~23 MB,
  ~60 us of link per step.
- **Determinism**: prod vs prod (two nodes) greedy identical; prod vs skip
  differs, but so does the hot set. Equal-residency control: profile capped to
  3,100 hot per GPU (`agent_space/profiles/glm53-w4a16-agentic-3239-r2000-
  cap3100.json` + VLLM_TIERED_MOE_PROFILE_CAP=1), arms eqbase / eqskip.
- **Staging cost, three versions** (same node jpbo-068-47, rank-0 trace,
  FlashMLA layer 0 -> 77 span per step):

| version | staging per group | effect on the main stream | span vs prod |
|---|---|---|---|
| v1 sort-based plan, forked before the anchor's FlashMLA | ~75 us of sort + elementwise kernels | anchor FlashMLA 18 -> 28 us, staging layers 245 -> 281 us | (different node) |
| v2 Triton claim / remap / gather (16 CTAs), forked after the anchor's FlashMLA | 38.8 us span, never late (slack >= 97 us) | claim overlaps the o_proj GEMM (4 -> 20 us): staging layers +15 us, skip layers +2-6 us | +0.29 ms at equal hot, -0.03 ms with +178 hot |
| v3 one-CTA claim / remap, 4-CTA gather, reserved map once per step (vllm c53f7c44c6) | | | running |

Step time from the bench rows (v2, same node, small samples): prod 22.37
ms/step (606 steps), skip 22.99 (359), skipsame 22.83 (1,221). On this
short-context agentic traffic the +178 hot experts buy only ~0.3 ms (the
replay's 0.97-1.22 ms was on live Claude Code traffic, with more cold
traffic), so the copy has to be close to free to win here.

## v4, drafter fp8 and the combined config (2026-10-06 night / 10-07)

**Skip-KV staging v4** (vllm 8ed7e0866f: mark with plain stores, compact with
one counter add per CTA, remap, 8-CTA gather): 23-25 us per group, never late
(slack >= 97 us), main-path kernels at prod speed. v3 (one-CTA claim) was
worse: 201 us per group, slack 0.5 us. The real v2 cost was ~600 adds on one
shared counter, not CTA count.

**Exactness**: prod vs prod greedy differs across runs too (one run differs at
the first token), so greedy identity is not a usable end-to-end check. Kernel
level: bit-exact (tests); FlashMLA and the KV write on Grace vs HBM are
bit-identical. Model level: GSM8K 200 prod 91.5% vs skip 92.5% (ab5, same node).

**Drafter fp8** (DRAFT_KV_DTYPE=fp8, DRAFT_QUANT=fp8_per_channel in serve.sh):
online PTPC quantization at load (per-output-channel weight scales, dynamic
per-token activation scales, CUTLASS fp8 scaled-mm: W8A8) on o_proj, MLP and
fc; qkv_proj stays bf16 (DFlash's context-KV precompute slices its raw rows),
conv kernel projections and candidate selector are unquantized by DFlash.
Needed vllm e3617a0248 (callable hf_overrides in get_quant_config),
d5bbc36d93 (resolve online shorthands for the draft ModelConfig), d78ce98864
(qkv_proj). fp8 drafter KV halves the drafter cache (2.29 -> 1.15 GiB per
GPU), which the planner turns into +59 hot experts; fp8 weights add none (the
GLM planner does not charge drafter weights).

| arm (32 agentic requests) | accepted/drafted | tokens/step | vs prod, matched requests |
|---|--:|--:|--:|
| prod (dfbase) | 0.334 | 3.34 | |
| fp8 drafter KV (dfkv) | 0.360 | 3.52 | +0.06 +- 0.28 |
| fp8 drafter weights (dfw4) | 0.369 | 3.59 | +0.03 +- 0.26 |
| both (dfboth4) | 0.350 | 3.45 | +0.11 +- 0.25 |

No acceptance drop detectable at this sample size.

**Combined config** (skip_host_uva + fp8 drafter KV + fp8 drafter weights,
skipkv-combo2, same node as its prod arm, 32 requests each): 3,418-3,420 hot
experts per GPU served vs 3,180 (+238); acceptance 0.373 vs 0.355 (+0.07 +-
0.10 tokens/step, matched). Step time, joint fit over both arms' requests
(>= 40 steps, step_ms ~ tokens/step + ctx + arm): **-0.13 +- 0.32 ms/step**,
not significant (skip tier alone on ab5: +0.33 +- 0.37). Four long matched
requests: -0.53 +- 0.12 ms/step, but the combo arm reached them at lower
context. Traces at similar context (prod 5.3K, combo 6.0K): target forward
21.09 -> 19.38 ms (MoE kernels 9.10 -> 6.86 ms), but the time between target
forwards (drafter, sampling) 2.30 -> 3.77 ms under the profiler: the fp8
drafter path launches more small kernels (per-token activation quant) and the
profiler inflates launches, so the unprofiled bench numbers are the measure.

So on this agentic traffic (the profile's own workload, low cold share) the
combined config is speed-neutral within +-0.3 ms; the replay predicts its gain
on live Claude Code traffic (more cold reads). Next: a fixed-prompt long-decode
probe with per-step server timing (VLLM_STEP_TRACE_FILE) to cut the noise,
and an unprofiled drafter-time comparison to see whether fp8 W8A8 drafting is
actually faster than bf16 at 8 tokens.
