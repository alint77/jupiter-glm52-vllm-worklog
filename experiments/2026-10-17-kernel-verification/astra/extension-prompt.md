# Design review: extending SentryWard to verify a Hopper cross-CTA MoE decode kernel

**Do not run any commands** (your sandbox cannot start on this machine). Reason from the material below.

Background: SentryWard (github.com/gpu-verifier/sentryward @60a1ebc) is a PTX verifier that concretely executes one launch, builds an event graph (program order + mbarrier / named-barrier / collective sync), checks mbarrier well-synchronization and happens-before data races. Immediate visibility is a fixed premise; no fences, no release/acquire, no global atomics (rejected at lowering), only mbarrier try_wait polling is recognized. We already got it to verify a simpler kernel of ours (skinny GEMM: TMA ring + mma.sync) on sm_90a PTX and it caught planted ring bugs that compute-sanitizer missed.

Now we want to verify `layer_kernel` of our TP-sliced tiered MoE decode kernel (full source excerpt below). I wrote a staged extension design (EXTENSION.md below). Please review it hard:

1. **Memory-model soundness (PTX ISA memory model, sm_90):** are the proposed sync rules right? Specifically: release/acquire at .gpu scope; `membar.gl`/`__threadfence` treated as fence.sc acting as both release and acquire fence with relaxed atomics; acq_rel RMW chains / release sequences (does PTX have release sequences like C++? what is the correct rule for `atom.acq_rel` last-arriver patterns and for `__threadfence(); atomicAdd(relaxed)`); red.add vs red.add not a race; an acquire performed by one lane (scheduler) then handed to another warp (producer) through an mbarrier arrive/wait at CTA scope, followed by `fence.proxy.async.global` and a bulk copy reading data released by another CTA — is that chain valid? Is `fence.proxy.async.global` in the producer the right/required fence, and does placing it in the *consuming* thread (not the writer) suffice? Same for `fence.proxy.async.shared::cta` before refilling a stage (generic reads of the previous use then TMA write): needed or not?
2. **Is the proposed S2 (pinned modification order + fixpoint + hb-consistency check) a sound way to get a meaningful, schedule-conditional verdict?** What can go wrong (e.g. the order affecting which CTA does the last-arriver work, which affects hb, which affects feasibility; claims whose values gate termination; the one-shot ready probe)? Is there a better abstraction (e.g. symbolic claims, or checking per-group invariants instead of per-execution)?
3. **Kernel bugs**: while reading the kernel against this protocol, do you see any actual synchronization bug or fragile assumption in `layer_kernel` (ordering of y13 red.adds vs the finisher's acq_rel RMW and activate/zero; hand_off barrier generations; ready probe hand-off; done_s fences; stage reuse/fence placement; K_END paths; steal path)? Cite lines.
4. **Prioritization / alternatives:** is GRID=1 first a good plan? What minimum subset gives the most bug-finding value per engineering hour? Would a Dartagnan/herd7-style litmus model of the cross-CTA protocol (extracted by hand) be a better use of time for the cross-CTA part, with SentryWard only for the intra-CTA rings? Estimate effort per stage if you can.
5. Any gap in "What this still won't cover" that matters more than listed.

Output: ranked findings (severity, claim, reasoning, concrete recommendation), then a recommended plan.

## EXTENSION.md
```markdown
# Extending SentryWard to the sliced decode MoE kernel (draft design)

Target: `layer_kernel` in
`vllm/model_executor/layers/fused_moe/tiered_decode/sliced_decode.cu:1027-1527`.
The goal is the same kind of result as skinny_gemm: well-synchronized and
race-free for one concrete small launch, conditional on stated pinned choices.

## What the kernel synchronizes

Each CTA has 352 threads in four warp roles:

- 8 consumer warps;
- a producer warp, which issues TMA and bulk copies;
- a scheduler warp, which claims work groups;
- a finisher warp, which completes w13 entries.

**Inside a CTA:**

- four mbarrier rings:
  - `full`/`empty`: TMA stage ring, 4 stages;
  - `sq_full`/`sq_empty`: scheduler → producer FIFO, 2 slots;
  - `hq_full`/`hq_empty`: consumers → finisher, 4 slots;
- `bar.sync 1` among the consumers (`consumer_sync`);
- `fence.proxy.async.shared::cta` before reusing a stage.

**Across CTAs, all through `Workspace* ws` in global memory:**

1. **Work claims:** the scheduler's lane 0 runs `atomicAdd(&ws->next[q], 1)`.
   The return value drives all of the CTA's control flow. When its own tier
   runs out, a CTA steals from the other tier once.
2. **w13 completion:** the finisher runs
   `atom.acq_rel.gpu.add(&done13[q][ei], nch)`. The last arriver:
   - reads y13 (accumulated with `red.add.f32` by consumers in every CTA);
   - writes x2 (`activate_routes`);
   - does `st.release.gpu ready[q][ei] = epoch`;
   - zeroes the y13 rows.
3. **w2 producers wait on the entry's ready flag:**
   `while (ld.acquire.gpu ready != epoch) nanosleep`, then
   `fence.proxy.async.global`, then a bulk copy (async proxy) of the x2 rows
   written by another CTA's finisher.
   - The scheduler also issues a one-shot `ld.acquire` probe of the flag.
   - If the probe sees `epoch`, the producer skips its spin. The acquire is
     handed over through the `sq` mbarrier, but the proxy fence still runs.
4. **Shared expert:**
   - consumers run `red.add` into y13s;
   - then `__threadfence`, `bar.sync`, and thread 0's
     `__threadfence; atomicAdd(done_s)` decide the last flusher;
   - the last flusher runs `__threadfence`, `activate_shared` (writes x2s),
     `fence.proxy.async`, `__threadfence`, `bar.sync`, then
     `st.release ready_s`;
   - the producers then wait on `ready_s` as in item 3.
5. **Outputs:** `red.add.f32` into `y` from every CTA. `finalize_kernel` and
   `route_prep_kernel` are separate kernels chained with PDL
   (`griddepcontrol`).

## Gaps in SentryWard 60a1ebc

| # | Gap | Evidence | Used by |
|---|---|---|---|
| G1 | Any global atomic add is rejected during lowering | `lowering.cpp:1486` `atomic_add_memory_semantics_unsupported`; event IR has no read-modify-write kind (`event_ir/event.h`) | 1, 2, 4, 5 |
| G2 | No release/acquire, no fences; immediate visibility is a fixed premise | `docs/assumptions.md`; the only orderings are program order, mbarrier, named barriers and collectives | 2, 3, 4 |
| G3 | The only polling it recognizes is mbarrier `try_wait`; a flag spin is "unmodeled polling" | `docs/assumptions.md` (bounded loops section) | 3, 4 |
| G4 | Control flow depends on global memory written by `route_prep` (`counts`, `lists[q][ei]` with `ntok/local/tok/route/wt`, `epoch`, `T`). `shape_arrays` only covers read-only tables behind a parameter pointer | `docs/input-json.md` | all |
| G5 | Proxies are not modeled, so a missing `fence.proxy.async` (generic write → async-proxy read) is accepted | no proxy concept in the IR | 3, 4 |
| G6 | Scale: `GRID=132` is a compile-time constant; 352 threads; HIDDEN=6144 gives 96 w13 + 48 w2 units per entry | `sliced_decode.cu:32-74` | all |
| G7 | Small issues | see below | |

G7 in detail:

- `prefetch.tensormap` clobbers its operand register (found on skinny);
- `bfi` is not in the concrete interpreter;
- `and` on a pointer is not followed;
- `nanosleep`;
- `griddepcontrol`;
- the int4/mxfp4 decode compute (`lop3`, `prmt`). Its values don't affect
  control or addresses, but it must not block execution.

## Proposed extension, staged

### S0: a verification build of the kernel, without changing the protocol

Under `-DTD_VERIFY`, the following become macros:

- GRID (1, 2 or 3);
- HIDDEN and INTER, shrunk while keeping every `static_assert`. In particular
  `R0CH > STAGES` must hold, so CHUNKS0 must be ≥ 5, e.g. HIDDEN=3072.
  INTER is reduced with TILES0/TILES1 consistent;
- MAX_TOKENS = 32.

Two routings: one routed expert per tier and no shared expert; and one routed
expert plus the shared expert.

### S1: concrete global memory image

New input `global_image`: typed records at (object, byte offset) with values,
marked `read_only` or `initial`.

- **Source:** run `route_prep_kernel` on the GPU for the chosen routing and dump
  the Workspace header.
- **Read-only:** `counts`, `lists`, `epoch`, `T`, `xs13`, `live`.
- **Initial:** `next = 0`, `done* = 0`, `ready = epoch - 1`, `y* = 0`.

Read-only values feed exact execution, so control and addresses become static.

### S2: global atomics with a pinned modification order

New event kinds:

- `AtomicRmw(addr, op, operand, order ∈ {relaxed, acq_rel}, scope)`;
- `AtomicWrite` for `red.*` (no return value);
- `ReleaseStore`, `AcquireLoad`;
- `Fence(kind ∈ {sc/acq_rel membar.gl, proxy.async.{global, shared}})`.

**Return values:**

- An input `atomic_order` policy fixes the modification order per location:
  `cta_round_robin`, `cta_major`, `seeded_random`, or an explicit list.
  RMW return values follow from that order.
- For claims (+1), CTA b's k-th claim returns a value computable from the
  policy alone.
- For counters with variable increments (`done13 += nch`), use a two-pass
  fixpoint: execute, collect the increments, recompute the returns, and
  re-execute until stable.

**Feasibility check:** after the happens-before (hb) graph is built, the chosen
modification order must not contradict hb. That means no RMW i ordered by hb
before an RMW j that comes earlier in the order. Otherwise report
`infeasible_atomic_order`. The verdict is conditional on the order, like
`branch_decisions`.

**Memory checker:**

- two atomic accesses to the same location at gpu scope never race;
- an atomic access against a plain access races unless ordered by hb.

**Synchronization (scoped C++-style subset of the PTX memory model):**

- A release (store, RMW, or release fence followed in program order by a
  relaxed RMW/store) synchronizes with an acquire (load, RMW, or relaxed
  read followed in program order by an acquire fence). The acquire must read
  from it or from a later RMW in its release sequence.
- `membar.gl` / `__threadfence` = `fence.sc.gpu`, which acts as both an
  acquire and a release fence.

### S3: normalizing flag polling

Recognize the loop

`L: ld.acquire.gpu r,[a]; setp.ne r, v; @p { nanosleep; } bra L`

and turn it into an `AwaitValue(a, v)` event. It resolves to the unique release
write of `v` to `a` in the execution and adds a synchronizes-with edge.
Diagnostics:

- `await_no_writer`: hang;
- `await_writer_after_waiter`: hb cycle, i.e. deadlock;
- `await_ambiguous_writers`.

The scheduler's one-shot probe becomes a load whose value is pinned
(`probe_decisions`, like `branch_decisions`):

- run once with the probe false (the producer spins);
- run once with the probe true where that is feasible: the writer must not be
  ordered after the probe by hb. This exercises the acquire handed over
  through the mbarrier.

### S4: proxy model (opt-in assumption `model_async_proxy`)

For a generic write W and an async-proxy read R of the same bytes (bulk/TMA
source in global memory, or a shared-memory TMA store source), require an hb
path W → … → F → R, where F is a `fence.proxy.async` of the matching state
space, placed in program order before R in R's issuing thread. Otherwise
report `missing_proxy_fence`.

The WAR direction (generic read, then async write after an mbarrier
handoff) stays ordered by the mbarrier alone, as in CUTLASS practice.

### S5: small items

- Fix the `prefetch.tensormap` clobber.
- Add `bfi`/`bfe`/`prmt`/`lop3` to the concrete interpreter, or treat them as
  opaque when they don't affect control or addresses.
- Make `nanosleep` a no-op.
- Make `griddepcontrol.wait`/`launch_dependents` no-ops for analysis of one
  kernel; the image stands in for the previous kernel's writes.

## Order of work

1. **GRID=1** with S0 + S1 + S2-lite + S3 + S5. With one CTA, every RMW on a
   location is issued by one lane, so the modification order is just program
   order and no fixpoint is needed. S2-lite =
   - atomic-write accesses (`red`) that don't race each other;
   - an RMW executed with sequential semantics;
   - release/acquire flag synchronization.

   This verifies:
   - all four rings;
   - y13 accumulation vs the last-arriver read and the zeroing;
   - the x2 write → ready → producer bulk read (same CTA, through global
     memory);
   - the shared-expert chain.
2. **GRID=2..3** with full S2 (order policies, fixpoint, feasibility) and
   several seeds.
3. **S4** proxy checking.

Positive controls for each stage, planted in the verification build:

- drop `st_release` / use a relaxed store;
- drop `fence.proxy.async`;
- read y13 before the `done13` RMW;
- `zero_routes` before `activate_routes`;
- skip the `ready` wait when `pre_ready` is wrong;
- an off-by-one in `sq`/`hq` parity.

## What this still won't cover

- All schedules: only the pinned or sampled modification orders and probe
  outcomes are covered.
- Real HIDDEN/INTER/GRID.
- `route_prep` and `finalize`, and the PDL chain between kernels.
- Reuse across launches (`epoch` wraparound, the zero-on-exit contract
  across calls).
- The PTX → SASS gap.
```

## sliced_decode.cu excerpts (with line numbers)
### constants, structs (1-175)
```cuda
     1	// SPDX-License-Identifier: Apache-2.0
     2	// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
     3	//
     4	// Decode-step MoE for TP-sliced tiered INT4 experts on sm_90a (GLM-5.3 W4A16:
     5	// symmetric INT4, bf16 group-32 scales, 6144 hidden, top-8).
     6	//
     7	// Every GPU holds a 512-wide slice of every routed expert's intermediate
     8	// dimension (gate / up rows [512 r, 512 r + 512), the matching down columns):
     9	// hot slices in HBM, cold slices in its own Grace memory, both in Marlin's
    10	// layout. The all-reduce after the MoE sums the four partial outputs.
    11	//
    12	// One persistent kernel per layer (route_prep -> layer_kernel -> finalize,
    13	// chained with programmatic dependent launch). layer_kernel is warp
    14	// specialized. A scheduler warp claims groups of 128-row x 512-K units from
    15	// dynamic per-tier queues (shared-expert w13, routed w13, shared-expert w2,
    16	// routed w2; idle CTAs steal across tiers) and hands them through a small
    17	// smem FIFO to a producer warp, which streams them with TMA into a 4-stage
    18	// smem ring. Eight consumer warps decode INT4 to exact f16 (one shift, four
    19	// lop3, four hfma2 per word) and run mma.sync m16n8k16 with fp32
    20	// accumulation. Routed w13 partials go to y13 through fp32 reds; a finisher
    21	// warp counts each finished w13 group, and the CTA that completes an entry
    22	// applies silu * up and publishes the entry, whose w2 units then run. The
    23	// scheduler claims the next group only after the producer has issued a w13
    24	// or shared w2 group, so a CTA never holds w13 work that others could start.
    25	// The shared expert (bf16 TP slice) is computed in the same kernel, so a side
    26	// stream is not needed. The output is routed_scale * routed + shared, the MoE's
    27	// final partial sum. Numerics match Marlin's up to fp32 summation order.
    28	//
    29	// C ABI (no torch headers): td_workspace_bytes(), td_forward(...).
    30	
    31	#include <cuda.h>
    32	#include <cuda_bf16.h>
    33	#include <cuda_fp16.h>
    34	#include <cuda_runtime.h>
    35	#include <cstdio>
    36	
    37	#include <cstddef>
    38	#include <cstring>
    39	#include <cstdint>
    40	
    41	namespace tiered_decode {
    42	
    43	// The 4-way intermediate slice of every expert: INTER = 512 per GPU. w13 and
    44	// w2 units are both 128 rows x 512 of K (w13: one K chunk, the 12 chunks of a
    45	// tile accumulate in registers; w2: all of K): 32 KB of weights and 4 KB of
    46	// bf16 scales each.
    47	#ifndef TD_STAGES
    48	  #define TD_STAGES 4  // 5 fits but measured slower
    49	#endif
    50	#ifndef TD_HQ
    51	  #define TD_HQ 4  // consumer -> finisher handoff slots
    52	#endif
    53	#ifndef TD_ACTK
    54	  #define TD_ACTK 4
    55	#endif
    56	#ifndef TD_SQ
    57	  #define TD_SQ 2  // scheduler -> producer FIFO depth (groups)
    58	#endif
    59	#ifndef TD_MAX_TOKENS
    60	  #define TD_MAX_TOKENS 64  // tokens per call
    61	#endif
    62	#ifndef TD_COLD_CTAS
    63	  #define TD_COLD_CTAS 16
    64	#endif
    65	constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
    66	constexpr int MAX_TOK = 8, MAX_TOKENS = TD_MAX_TOKENS,
    67	              MAX_ROUTES = MAX_TOKENS * TOPK, MAX_LIST = MAX_ROUTES;
    68	constexpr int R0 = 128, R1 = 128;                // rows per w13 / w2 unit
    69	constexpr int CK0 = 512;                         // w13 K chunk
    70	constexpr int KT0 = CK0 / 16, KT1 = INTER / 16;  // k16 rows per unit
    71	constexpr int G0 = CK0 / 32, G1 = INTER / 32;    // scale groups per unit
    72	constexpr int TILES0 = 2 * INTER / R0, CHUNKS0 = HIDDEN / CK0;
    73	constexpr int UNITS0 = TILES0 * CHUNKS0;  // w13 units per entry
    74	constexpr int TILES1 = HIDDEN / R1;       // w2 units per entry
    75	constexpr int W_BYTES = R0 * CK0 / 2, S_BYTES = G0 * R0 * 2;
    76	static_assert(W_BYTES == R1 * INTER / 2 && S_BYTES == G1 * R1 * 2,
    77	              "w13 and w2 units share the stage layout");
    78	enum Fmt : int { MXFP4 = 0, INT4 = 1 };
    79	constexpr int XROW_BYTES0 = CK0 * 2, XROW_BYTES1 = INTER * 2;
    80	constexpr int XROWS = MAX_TOK;
    81	constexpr int XROW_STRIDE =
    82	    XROW_BYTES0 + 64;  // token rows g, g+1 on disjoint banks
    83	static_assert(XROW_BYTES0 == XROW_BYTES1, "w13 and w2 rows are the same size");
    84	// an x2 row in the workspace: INTER halves, then its fp32 scale (xs2), so one
    85	// bulk copy brings both and the producer has no dependent xs2 load
    86	constexpr int X2_LD = INTER + 8, X2_COPY = XROW_BYTES1 + 16;
    87	static_assert(X2_COPY <= XROW_STRIDE, "x2 row + scale fit a stage row");
    88	// shared expert (bf16, this GPU's 512-wide TP slice of it): units of 256 rows
    89	// x 64 of K, 128 B swizzled, for all T tokens
    90	constexpr int RS = 256, CKS = 64;
    91	constexpr int TILES_S0 = 2 * INTER / RS, CHUNKS_S0 = HIDDEN / CKS;
    92	constexpr int TILES_S1 = HIDDEN / RS, CHUNKS_S1 = INTER / CKS;
    93	constexpr int UNITS_S0 = TILES_S0 * CHUNKS_S0, UNITS_S1 = TILES_S1 * CHUNKS_S1;
    94	static_assert(RS * CKS * 2 == W_BYTES, "a shared unit fills the weight slot");
    95	constexpr int XS_BYTES =
    96	    CKS * 2;  // one token's activation slice per shared unit
    97	// stages start on 1 KB boundaries (128 B swizzle)
    98	constexpr int STAGE_BYTES =
    99	    (W_BYTES + S_BYTES + XROWS * XROW_STRIDE + 1023) / 1024 * 1024;
   100	static_assert(MAX_TOKENS * XS_BYTES <= STAGE_BYTES - W_BYTES,
   101	              "shared rows fit");
   102	static_assert(MAX_TOKENS % 32 == 0, "token tiles; scheduler lanes per token");
   103	constexpr int NT_MAX = MAX_TOKENS / 8;  // shared expert mma token tiles
   104	constexpr int STAGES = TD_STAGES;
   105	constexpr int CONSUMER_WARPS = 8;
   106	constexpr int THREADS = (CONSUMER_WARPS + 3) * 32;  // + producer, scheduler,
   107	                                                    // finisher
   108	constexpr int SMEM_HEAD = 128;
   109	static_assert(STAGES <= 8, "barrier head holds 8 stages");
   110	constexpr int SMEM_BYTES = SMEM_HEAD + 1024 + STAGES * STAGE_BYTES;
   111	static_assert(SMEM_BYTES <= 227 * 1024, "ring exceeds shared memory");
   112	constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
   113	constexpr int GRID = 132;
   114	constexpr int PREP_THREADS =
   115	    1024;  // route_prep block: 6 hidden elements per thread
   116	static_assert(
   117	    CONSUMER_WARPS == 8,
   118	    "consumer warps are 4 row blocks x 2 K halves (w13), 8 row blocks (w2)");
   119	
   120	struct Expert {
   121	  int local;  // index into the tier's tensors
   122	  int ntok;
   123	  int tok[MAX_TOK];    // token rows
   124	  int route[MAX_TOK];  // token * TOPK + k
   125	  float wt[MAX_TOK];   // router weight * routed_scale
   126	};
   127	
   128	// Per tier and projection: the weight and scale tensor maps. A weight map is
   129	// [E][K/16][N*2] int32 with a {128, 64, 1} box: one 64-row tile's 64 k16 rows,
   130	// 512 B each, landing contiguous in shared memory. A scale map is [E][K/32][N]
   131	// bytes with a {64, 32, 1} box.
   132	struct alignas(64) Tier {
   133	  CUtensorMap w[2];
   134	  CUtensorMap s[2];
   135	};
   136	
   137	// Device workspace, zeroed once at allocation; y13 and y are re-zeroed by
   138	// their last reader so every call finds them clean.
   139	struct Workspace {
   140	  Expert lists[2][MAX_LIST];  // hot, cold
   141	  int counts[2];
   142	  int live[MAX_ROUTES];  // route runs an expert on this GPU
   143	  float xs13[MAX_TOKENS];
   144	  float xs2[MAX_ROUTES];
   145	  alignas(128)
   146	      __half x13[MAX_TOKENS * HIDDEN];  // TMA sources: 16 B aligned at least
   147	  alignas(128) __half x2[MAX_ROUTES * X2_LD];
   148	  alignas(128) float y13[MAX_ROUTES * 2 * INTER];
   149	  alignas(128) float y[MAX_TOKENS * HIDDEN];
   150	  int done13[2]
   151	            [MAX_LIST];    // w13 chunks flushed per entry, zeroed by route_prep
   152	  int ready[2][MAX_LIST];  // epoch once the entry's activation rows are written
   153	  int epoch;               // bumped by route_prep every call
   154	  alignas(128) __nv_bfloat16 x13b[MAX_TOKENS * HIDDEN];  // shared expert input
   155	  alignas(128) __nv_bfloat16 x2s[MAX_TOKENS * INTER];    // its activation
   156	  alignas(128) float y13s[MAX_TOKENS * 2 * INTER];
   157	  int next[2];  // per tier: next group to claim (hot, cold), zeroed by
   158	                // route_prep
   159	  int done_s;   // shared w13 chunks flushed
   160	  int ready_s;  // epoch once x2s is written
   161	  int T;
   162	};
   163	static_assert(offsetof(Workspace, x13) % 16 == 0 &&
   164	                  offsetof(Workspace, x2) % 16 == 0,
   165	              "TMA alignment");
   166	
   167	struct Params {
   168	  Tier tier[2];
   169	  CUtensorMap
   170	      sw[2];  // shared expert w13 [2 * INTER][HIDDEN], w2 [HIDDEN][INTER]
   171	  Workspace* ws;
   172	  float shared_scale;
   173	  int has_shared;
   174	};
   175	
```
### PTX helpers (176-260, 350-356)
```cuda
  176  // ---------------------------------------------------------------- PTX helpers
  177  __device__ __forceinline__ uint32_t smem_u32(const void* p) {
  178    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
  179  }
  180  __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  181    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
  182                 "r"(count));
  183  }
  184  __device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  185    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
  186                     smem_u32(bar)),
  187                 "r"(bytes)
  188                 : "memory");
  189  }
  190  __device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  191    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
  192                 : "memory");
  193  }
  194  __device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
  195    uint32_t v;
  196    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
  197    return v;
  198  }
  199  __device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  200    uint4 v;
  201    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
  202                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
  203                 : "r"(a));
  204    return v;
  205  }
  206  __device__ __forceinline__ void mbar_wait_a(uint32_t bar, uint32_t parity) {
  207    asm volatile(
  208        "{\n .reg .pred p;\n WAITA_%=:\n "
  209        "mbarrier.try_wait.parity.shared::cta.b64 "
  210        "p, [%0], %1;\n"
  211        " @!p bra WAITA_%=;\n}\n" ::"r"(bar),
  212        "r"(parity)
  213        : "memory");
  214  }
  215  __device__ __forceinline__ void mbar_arrive_a(uint32_t bar) {
  216    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
  217                 : "memory");
  218  }
  219  __device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  220    asm volatile(
  221        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
  222        "p, [%0], %1;\n"
  223        " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
  224        "r"(parity)
  225        : "memory");
  226  }
  227  __device__ __forceinline__ void bulk_g2s(void* dst, const void* src,
  228                                           uint32_t bytes, uint64_t* bar) {
  229    asm volatile(
  230        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
  231        "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
  232        "l"(src), "r"(bytes), "r"(smem_u32(bar))
  233        : "memory");
  234  }
  235  __device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map,
  236                                         int c0, int c1, int c2, uint64_t* bar) {
  237    asm volatile(
  238        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
  239        "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
  240        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
  241        "r"(smem_u32(bar))
  242        : "memory");
  243  }
  244  __device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map,
  245                                         int c0, int c1, uint64_t* bar) {
  246    asm volatile(
  247        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
  248        "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
  249        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
  250        : "memory");
  251  }
  252  __device__ __forceinline__ void ldmatrix_x4(uint32_t* a, const void* p) {
  253    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
  254                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
  255                 : "r"(smem_u32(p)));
  256  }
  257  __device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a,
  258                                           uint32_t b0, uint32_t b1) {
  259    asm volatile(
  260        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
  350  __device__ __forceinline__ void pdl_wait() {
  351    asm volatile("griddepcontrol.wait;" ::: "memory");
  352  }
  353  __device__ __forceinline__ void pdl_release() {
  354    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  355  }
  356  __device__ __forceinline__ float e8m0(uint32_t byte) {
```
### cross-CTA helpers, flush, activation (726-924)
```cuda
  726  // ---------------------------------------------------------------- 2: the
  727  // layer, one persistent CTA per SM. CTAs [0, cold_ctas) run the cold tier,
  728  // the rest the hot tier. Each CTA runs its share of its tier's w13 units, then
  729  // its share of the w2 units. An entry's w13 chunks are counted as they flush;
  730  // the CTA whose flush completes an entry runs silu * up for its routes and
  731  // releases ready[entry]. A w2 unit's producer issues the weights first and
  732  // waits on ready only before loading the activation rows.
  733  __device__ __forceinline__ int cold_ctas_for(int n_hot, int n_cold) {
  734    if (n_cold == 0) return 0;
  735    return n_hot == 0 ? GRID : TD_COLD_CTAS;
  736  }
  737  __device__ __forceinline__ int ld_acquire(const int* p) {
  738    int v;
  739    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
  740                 : "=r"(v)
  741                 : "l"(p)
  742                 : "memory");
  743    return v;
  744  }
  745  __device__ __forceinline__ void st_release(int* p, int v) {
  746    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v)
  747                 : "memory");
  748  }
  749  __device__ __forceinline__ void fence_proxy_async() {
  750    asm volatile("fence.proxy.async.global;" ::: "memory");
  751  }
  752  // orders the consumers' generic reads of a released stage before the async
  753  // copies that refill it
  754  __device__ __forceinline__ void fence_proxy_async_smem() {
  755    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  756  }
  757  // predicated fire-and-forget add: no branch per element
  758  __device__ __forceinline__ void red_add_if(float* a, float v, bool p) {
  759    asm volatile(
  760        "{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q red.global.add.f32 [%0], "
  761        "%1;\n}" ::"l"(a),
  762        "f"(v), "r"(static_cast<int>(p))
  763        : "memory");
  764  }
  765  __device__ __forceinline__ void red_add_v4(float* a, float4 v) {
  766    asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};" ::"l"(a),
  767                 "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
  768                 : "memory");
  769  }
  770  __device__ __forceinline__ void sts_f32(uint32_t a, float v) {
  771    asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v));
  772  }
  773  __device__ __forceinline__ float4 lds_f4(uint32_t a) {
  774    float4 v;
  775    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
  776                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
  777                 : "r"(a));
  778    return v;
  779  }
  780  // One warp's 64 rows x ntok tokens of a routed unit (acc * per-token scale)
  781  // added into base + row(token) * stride: staged through the warp's smem
  782  // scratch [token][68] (conflict-free), then vector reds over contiguous rows.
  783  constexpr int SCR_LD = 68;
  784  __device__ __forceinline__ void flush_rows(float (*acc)[4], const float* fs,
  785                                             uint32_t scr, float* base,
  786                                             const int* rows4, int stride,
  787                                             int ntok, int g, int tq, int lane) {
  788  #pragma unroll
  789    for (int mb = 0; mb < 4; ++mb)
  790  #pragma unroll
  791      for (int i = 0; i < 4; ++i) {
  792        sts_f32(
  793            scr + ((2 * tq + (i & 1)) * SCR_LD + mb * 16 + g + (i >> 1) * 8) * 4,
  794            acc[mb][i] * fs[i & 1]);
  795        acc[mb][i] = 0.f;
  796      }
  797    __syncwarp();
  798  #pragma unroll
  799    for (int j = 0; j < MAX_TOK / 2; ++j) {  // token c = lane / 16 + 2 j
  800      const int k = lane + 32 * j;
  801      if (k < ntok * 16) {
  802        const int c = k >> 4, r4 = k & 15;
  803        const float4 v = lds_f4(scr + (c * SCR_LD + r4 * 4) * 4);
  804        red_add_v4(base + static_cast<size_t>(rows4[j]) * stride + r4 * 4, v);
  805      }
  806    }
  807    __syncwarp();
  808  }
  809  __device__ __forceinline__ void consumer_sync() {
  810    asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
  811  }
  812  
  813  // silu(gate) * up for one entry's routes, one warp per route: f16 fragments
  814  // of x2 with the row's power-of-two scale; the y13 rows are zeroed for the
  815  // next call.
  816  __device__ __forceinline__ void activate_route(Workspace* ws, int r, int lane) {
  817    float* __restrict__ yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
  818    float a[INTER / 32], m = 0.f;
  819  #pragma unroll
  820    for (int q = 0; q < INTER / 32; ++q) {
  821      const float g = __ldcg(yr + q * 32 + lane),
  822                  u = __ldcg(yr + INTER + q * 32 + lane);
  823      a[q] = __fdividef(g, 1.f + __expf(-g)) * u;
  824      m = fmaxf(m, fabsf(a[q]));
  825    }
  826    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  827    float scale;
  828    const float inv = row_scale(m, &scale);
  829    if (lane == 0) ws->xs2[r] = scale;
  830    __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
  831    if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
  832  #pragma unroll
  833    for (int q = 0; q < INTER / 32; ++q) {
  834      out[frag_slot(q * 32 + lane)] = __float2half_rn(a[q] * inv);
  835      yr[q * 32 + lane] = 0.f;
  836      yr[INTER + q * 32 + lane] = 0.f;
  837    }
  838  }
  839  
  840  // activate_route for the n routes in lanes [0, n) of routel, K at a time
  841  template <int K>
  842  __device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
  843                                                  int n, int lane,
  844                                                  uint64_t* tl = nullptr) {
  845    for (int r0 = 0; r0 < n; r0 += K) {
  846      float a[K][INTER / 32], m[K];
  847      float* yr[K];
  848  #pragma unroll
  849      for (int j = 0; j < K; ++j) {
  850        const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
  851        m[j] = 0.f;
  852        if (r0 + j < n) {
  853          yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
  854  #pragma unroll
  855          for (int q = 0; q < INTER / 32; ++q) {
  856            const float g = __ldcg(yr[j] + q * 32 + lane),
  857                        u = __ldcg(yr[j] + INTER + q * 32 + lane);
  858            a[j][q] = __fdividef(g, 1.f + __expf(-g)) * u;
  859            m[j] = fmaxf(m[j], fabsf(a[j][q]));
  860          }
  861        }
  862      }
  863  #pragma unroll
  864      for (int j = 0; j < K; ++j) {
  865        if (r0 + j >= n) break;
  866        for (int o = 16; o; o >>= 1)
  867          m[j] = fmaxf(m[j], __shfl_xor_sync(0xffffffffu, m[j], o));
  868        float scale;
  869        const float inv = row_scale(m[j], &scale);
  870        const int r = __shfl_sync(0xffffffffu, routel, r0 + j);
  871        if (lane == 0) ws->xs2[r] = scale;
  872        __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
  873        if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
  874  #pragma unroll
  875        for (int q = 0; q < INTER / 32; ++q) {
  876          out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
  877        }
  878      }
  879    }
  880  }
  881  
  882  // zero the y13 rows of the n routes in lanes [0, n) of routel for the next
  883  // call
  884  __device__ __forceinline__ void zero_routes(Workspace* ws, int routel, int n,
  885                                              int lane) {
  886    for (int j = 0; j < n; ++j) {
  887      const int r = __shfl_sync(0xffffffffu, routel, j);
  888      float4* __restrict__ yr =
  889          reinterpret_cast<float4*>(ws->y13 + static_cast<size_t>(r) * 2 * INTER);
  890  #pragma unroll
  891      for (int q = 0; q < 2 * INTER / 128; ++q)
  892        yr[q * 32 + lane] = make_float4(0.f, 0.f, 0.f, 0.f);
  893    }
  894  }
  895  
  896  // The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
  897  // y13s is zeroed for the next call.
  898  __device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
  899                                                  int lane) {
  900    for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
  901      float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
  902      __nv_bfloat16* __restrict__ out =
  903          ws->x2s + static_cast<size_t>(tok) * INTER;
  904  #pragma unroll
  905      for (int q = 0; q < INTER / 32; ++q) {
  906        const int k = q * 32 + lane;
  907        const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
  908        out[frag_slot(k)] =
  909            __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
  910        yr[k] = 0.f;
  911        yr[INTER + k] = 0.f;
  912      }
  913    }
  914  }
  915  
  916  // One routed unit's 16 group steps for this warp. PH 0 (w13): 4 row blocks x
  917  // 2 K halves of a 64-row x 1024 box; PH 1 (w2): 8 row blocks of a 128-row x
  918  // 512 box. All smem offsets are immediates off two per-warp bases.
  919  // One routed unit for this warp: all 4 row blocks of a 64-row tile over 4
  920  // group steps of K, so each lane's 16 B of a Marlin k16 row (its 4 row blocks)
  921  // is one conflict-free LDS.128, and each activation fragment feeds 4 MMAs.
  922  // w13 unit (64 rows x 1024): warp w takes K steps [4w, 4w + 4); w2 unit (128
  923  // rows x 512): warp w takes 64-row half w / 4, K steps [4 (w % 4), + 4).
  924  template <int PH>
```
### queues + layer_kernel (966-1527)
```cuda
  966  
  967  // Work queues. Each tier's work is a list of groups, claimed in order with
  968  // one atomic by whichever CTA's producer is free: the shared expert's w13
  969  // (tile x 12 chunks), the routed w13 (entry x tile, 6 chunks), the shared w2
  970  // (tile, 8 chunks), the routed w2 (entry x 128-row unit). An entry's w13
  971  // groups go out early and to many CTAs at once, so its w2 units rarely wait;
  972  // the queue ends on 1-unit groups, so CTAs finish within a unit of each
  973  // other. A CTA whose own tier's queue is empty takes the other tier's.
  974  enum Kind : int { K_S0 = 0, K_R0 = 1, K_S1 = 2, K_R1 = 3, K_END = 15 };
  975  constexpr int SCH0 = 12;               // shared w13 chunks per group
  976  constexpr int SG0 = CHUNKS_S0 / SCH0;  // groups per shared w13 tile
  977  static_assert(CHUNKS_S0 % SCH0 == 0, "shared w13 groups tile K");
  978  struct Group {
  979    int kind, x, c0,
  980        nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
  981  };
  982  // one claimed group, decoded, with its entry record (scheduler -> producer)
  983  struct SchedEntry {
  984    int kind, x, c0, nch, q, ei, t0, ntok, local, ready;
  985    int tok[MAX_TOK], route[MAX_TOK];
  986    float f[MAX_TOK];  // w13: the token's x13 scale; w2: the route weight
  987  };
  988  // routed w2 units per claim (consecutive tiles of one entry): GR1 above
  989  // GR1_T tokens, else 1. Larger claims amortize the producer's claim, record
  990  // and ready round trips (M=16/32 -5..7%); at M=8 they unbalance the tail.
  991  #ifndef TD_GR1
  992    #define TD_GR1 2
  993  #endif
  994  #ifndef TD_GR1_T
  995    #define TD_GR1_T 8
  996  #endif
  997  constexpr int GR1 = TD_GR1;
  998  static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
  999  // A routed w13 tile's 12 K chunks go out as R0S adjacent groups, so several
 1000  // CTAs share a late tile and its entry's activation is not stuck behind one
 1001  // CTA's 12 serial units (the w2 producers spin on it).
 1002  #ifndef TD_R0S
 1003    #define TD_R0S 1  // 2-4 measured slower: more w13 flushes
 1004  #endif
 1005  constexpr int R0S = TD_R0S, R0CH = CHUNKS0 / R0S;
 1006  static_assert(CHUNKS0 % R0S == 0, "w13 groups split the K chunks evenly");
 1007  // warps 1..7 run at most STAGES units ahead of warp 0 (the ring), so with
 1008  // more units than that between w13 flushes no warp can arrive at the handoff
 1009  // barrier for the next flush before warp 0 has synced on this one
 1010  static_assert(R0CH > STAGES, "w13 handoff barrier generations cannot overlap");
 1011  __device__ __forceinline__ int queue_len(int n, bool sh, int g1) {
 1012    return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
 1013           n * (TILES0 * R0S + TILES1 / g1);
 1014  }
 1015  __device__ __forceinline__ Group group_at(int gi, int n, bool sh, int g1) {
 1016    const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0 * R0S,
 1017              ns1 = sh ? TILES_S1 : 0;
 1018    if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
 1019    gi -= ns0;
 1020    if (gi < nr0) return {K_R0, gi / R0S, (gi % R0S) * R0CH, R0CH};
 1021    gi -= nr0;
 1022    if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
 1023    gi -= ns1;
 1024    return {K_R1, gi, 0, g1};
 1025  }
 1026  
 1027  __global__ void __launch_bounds__(THREADS, 1)
 1028      layer_kernel(const __grid_constant__ Params p) {
 1029    extern __shared__ __align__(128) unsigned char smem[];
 1030    uint64_t* full = reinterpret_cast<uint64_t*>(smem);
 1031    uint64_t* empty = full + STAGES;
 1032    // 1 KB aligned (128 B swizzle) by an offset into smem, so every load off it
 1033    // stays a shared-space LDS rather than a generic LD
 1034    unsigned char* ring =
 1035        smem + SMEM_HEAD +
 1036        ((1024u - ((smem_u32(smem) + SMEM_HEAD) & 1023u)) & 1023u);
 1037    __shared__ int4 desc[STAGES];
 1038    // per stage, routed units: each token's destination row (w13: its route's
 1039    // y13 row; w2: its token's y row) and the scale its flush applies
 1040    // (w13: xs13[token]; w2: the route weight, times the row's xs2 at the
 1041    // flush), written by the producer
 1042    __shared__ int sd_row[STAGES][MAX_TOK];
 1043    __shared__ float sd_f[STAGES][MAX_TOK];
 1044    __shared__ int s_last;
 1045    __shared__ SchedEntry sq[TD_SQ];
 1046    __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
 1047    __shared__ int4 hq_info[TD_HQ];  // (q, entry, chunks, end)
 1048    __shared__ uint64_t hq_full[TD_HQ], hq_empty[TD_HQ];
 1049    __shared__ __align__(16) float scr_all[CONSUMER_WARPS][MAX_TOK * SCR_LD];
 1050    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 1051    Workspace* ws = p.ws;
 1052    if (threadIdx.x == 0) {
 1053      for (int s = 0; s < STAGES; ++s) {
 1054        mbar_init(&full[s], 1);
 1055        mbar_init(&empty[s], CONSUMER_WARPS);
 1056      }
 1057      for (int k = 0; k < TD_SQ; ++k) {
 1058        mbar_init(&sq_full[k], 1);
 1059        mbar_init(&sq_empty[k], 1);
 1060      }
 1061      for (int k = 0; k < TD_HQ; ++k) {
 1062        mbar_init(&hq_full[k], CONSUMER_WARPS);
 1063        mbar_init(&hq_empty[k], 1);
 1064      }
 1065      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 1066    }
 1067    __syncthreads();
 1068    pdl_wait();  // route_prep's lists, rows and counters
 1069    pdl_release();
 1070    const int n_tier[2] = {ws->counts[0], ws->counts[1]};
 1071    const int epoch = ws->epoch, T = ws->T;
 1072    const bool sh = p.has_shared;
 1073    const int g1 = T > TD_GR1_T ? GR1 : 1;
 1074    const int len[2] = {queue_len(n_tier[0], sh, g1),
 1075                        queue_len(n_tier[1], false, g1)};
 1076    const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
 1077    const int own = static_cast<int>(blockIdx.x) < cold_ctas ? 1 : 0;
 1078  
 1079    if (warp == CONSUMER_WARPS + 2) {  // finisher warp
 1080      int k = 0;
 1081      uint32_t kph = 0;
 1082      while (true) {
 1083        if (lane == 0) mbar_wait(&hq_full[k], kph);
 1084        __syncwarp();
 1085        const int4 h = hq_info[k];
 1086        __syncwarp();
 1087        if (lane == 0) mbar_arrive(&hq_empty[k]);
 1088        if (++k == TD_HQ) {
 1089          k = 0;
 1090          kph ^= 1;
 1091        }
 1092        if (h.w) break;
 1093        const int q = h.x, ei = h.y, nch = h.z;
 1094        const Expert& e = ws->lists[q][ei];
 1095        const int n = e.ntok, routel = e.route[lane & (MAX_TOK - 1)];
 1096        int done = 0;
 1097        if (lane == 0) {
 1098          int old;
 1099          asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
 1100                       : "=r"(old)
 1101                       : "l"(&ws->done13[q][ei]), "r"(nch)
 1102                       : "memory");
 1103          done = old + nch == UNITS0;
 1104        }
 1105        done = __shfl_sync(0xffffffffu, done, 0);
 1106        if (done) {
 1107          __syncwarp();  // lane 0's acquire (the count) ordered before every
 1108                         // lane's y13 reads
 1109          activate_routes<TD_ACTK>(ws, routel, n, lane);
 1110          __syncwarp();
 1111          if (lane == 0) st_release(&ws->ready[q][ei], epoch);
 1112          zero_routes(ws, routel, n, lane);
 1113        }
 1114      }
 1115      return;
 1116    }
 1117    if (warp == CONSUMER_WARPS + 1) {  // scheduler warp
 1118      // token t's x13 scale in lane t % 32 of register t / 32
 1119      float xs13r[MAX_TOKENS / 32];
 1120  #pragma unroll
 1121      for (int j = 0; j < MAX_TOKENS / 32; ++j)
 1122        xs13r[j] = lane + 32 * j < T ? ws->xs13[lane + 32 * j] : 0.f;
 1123      int q = own, n = 0, k = 0;
 1124      bool stolen = false;
 1125      int gi = 0;  // this group's claim; the next one is issued before its loads
 1126      if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
 1127      gi = __shfl_sync(0xffffffffu, gi, 0);
 1128      while (true) {
 1129        if (n >= TD_SQ) {  // slot k is free once its last group is issued
 1130          if (lane == 0) mbar_wait(&sq_empty[k], ((n / TD_SQ) - 1) & 1);
 1131          __syncwarp();
 1132        }
 1133        SchedEntry& e = sq[k];
 1134        int gn = 0;
 1135        // no claim-ahead past a w13 or shared w2 group: its successor is
 1136        // claimed once the producer has issued all of its chunks
 1137        const bool heavy =
 1138            gi < (q ? len[1] : len[0]) &&
 1139            group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1).kind !=
 1140                K_R1;
 1141        if (lane == 0 && !heavy) gn = atomicAdd(&ws->next[q], 1);
 1142        if (gi >= (q ? len[1] : len[0])) {
 1143          bool steal = !stolen;
 1144          if (steal) {
 1145            stolen = true;
 1146            q ^= 1;
 1147            if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
 1148            gi = __shfl_sync(0xffffffffu, gi, 0);
 1149            continue;
 1150          }
 1151          if (lane == 0) e.kind = K_END;
 1152          __syncwarp();
 1153          if (lane == 0) mbar_arrive(&sq_full[k]);
 1154          break;
 1155        }
 1156        const Group gr =
 1157            group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1);
 1158        int ei = 0, t0 = 0;
 1159        if (gr.kind == K_R0) {
 1160          ei = gr.x / TILES0;
 1161          t0 = gr.x - ei * TILES0;
 1162        } else if (gr.kind == K_R1) {
 1163          if (g1 == 1) {
 1164            ei = gr.x / TILES1;
 1165            t0 = gr.x - ei * TILES1;
 1166          } else {
 1167            ei = gr.x / (TILES1 / GR1);
 1168            t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;
 1169          }
 1170        }
 1171        int rdy = 0;
 1172        if (gr.kind == K_R0 || gr.kind == K_R1) {
 1173          // one round trip: every field at once (lanes past ntok read a valid
 1174          // slot and are masked later)
 1175          const Expert& x = ws->lists[q][ei];
 1176          const int l7 = lane & (MAX_TOK - 1);
 1177          const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
 1178                    routel = x.route[l7];
 1179          const float wtl = x.wt[l7];
 1180          // the ready probe in flight with the record loads
 1181          if (lane == 0 && gr.kind == K_R1)
 1182            rdy = ld_acquire(&ws->ready[q][ei]) == epoch;
 1183          float xs = __shfl_sync(0xffffffffu, xs13r[0], tokl & 31);
 1184  #pragma unroll
 1185          for (int j = 1; j < MAX_TOKENS / 32; ++j) {
 1186            const float v = __shfl_sync(0xffffffffu, xs13r[j], tokl & 31);
 1187            if (tokl >> 5 == j) xs = v;
 1188          }
 1189          if (lane < MAX_TOK) {
 1190            e.tok[lane] = tokl;
 1191            e.route[lane] = routel;
 1192            e.f[lane] = gr.kind == K_R0 ? xs : wtl;
 1193          }
 1194          if (lane == 0) {
 1195            e.ntok = ntok;
 1196            e.local = local;
 1197          }
 1198        }
 1199        if (lane == 0) {
 1200          e.kind = gr.kind;
 1201          e.x = gr.x;
 1202          e.c0 = gr.c0;
 1203          e.nch = gr.nch;
 1204          e.q = q;
 1205          e.ei = ei;
 1206          e.t0 = t0;
 1207          // one probe: if the entry is ready, this acquire is handed to the
 1208          // producer by the FIFO barrier and its spin is skipped
 1209          e.ready = rdy;
 1210        }
 1211        __syncwarp();
 1212        if (lane == 0) mbar_arrive(&sq_full[k]);
 1213        if (heavy && lane == 0) {
 1214          mbar_wait(&sq_empty[k], (n / TD_SQ) & 1);
 1215          gn = atomicAdd(&ws->next[q], 1);
 1216        }
 1217        if (++k == TD_SQ) k = 0;
 1218        ++n;
 1219        gi = __shfl_sync(0xffffffffu, gn, 0);
 1220      }
 1221      return;
 1222    }
 1223    if (warp == CONSUMER_WARPS) {  // producer warp
 1224      if (lane == 0) {
 1225        for (int q = 0; q < 2; ++q)
 1226          for (int k = 0; k < 2; ++k) {
 1227            asm volatile("prefetch.tensormap [%0];" ::"l"(
 1228                             reinterpret_cast<uint64_t>(&p.tier[q].w[k]))
 1229                         : "memory");
 1230            asm volatile("prefetch.tensormap [%0];" ::"l"(
 1231                             reinterpret_cast<uint64_t>(&p.tier[q].s[k]))
 1232                         : "memory");
 1233          }
 1234        if (sh)
 1235          for (int k = 0; k < 2; ++k)
 1236            asm volatile("prefetch.tensormap [%0];" ::"l"(
 1237                             reinterpret_cast<uint64_t>(&p.sw[k]))
 1238                         : "memory");
 1239      }
 1240      int last_ready = -1, it = 0, k = 0;
 1241      uint32_t kph = 0;
 1242      const int gi = 0;
 1243      (void)gi;
 1244      while (true) {
 1245        if (lane == 0) mbar_wait(&sq_full[k], kph);
 1246        __syncwarp();
 1247        const SchedEntry& e = sq[k];
 1248        if (e.kind == K_END) break;
 1249        const Group gr = {e.kind, e.x, e.c0, e.nch};
 1250        const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = e.ready;
 1251        int ntok = 0, local = 0, tokl = 0, routel = 0;
 1252        float fl = 0.f;
 1253        if (gr.kind == K_R0 || gr.kind == K_R1) {
 1254          const int l7 = lane & (MAX_TOK - 1);
 1255          ntok = e.ntok;
 1256          local = e.local;
 1257          tokl = e.tok[l7];
 1258          routel = e.route[l7];
 1259          fl = e.f[l7];
 1260        }
 1261        const Tier& tr = p.tier[q];
 1262        for (int ci = 0; ci < gr.nch; ++ci, ++it) {
 1263          const int s = it % STAGES, c = gr.c0 + ci;
 1264          unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
 1265          if (lane == 0 && it >= STAGES)
 1266            mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
 1267          __syncwarp();
 1268          fence_proxy_async_smem();
 1269          const int hdr = gr.kind | q << 4 | (ci == 0) << 8 |
 1270                          (ci == gr.nch - 1) << 9 | gr.nch << 16;
 1271          if (gr.kind == K_R0) {
 1272            if (lane < ntok) {
 1273              sd_row[s][lane] = routel;
 1274              sd_f[s][lane] = fl;
 1275            }
 1276            __syncwarp();
 1277            if (lane == 0) {
 1278              desc[s] = make_int4(hdr, ei, t0, ntok);
 1279              mbar_expect_tx(&full[s],
 1280                             W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
 1281              tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
 1282              tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
 1283            }
 1284            __syncwarp();
 1285            if (lane < ntok && lane < XROWS)
 1286              bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
 1287                       ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
 1288                       XROW_BYTES0, &full[s]);
 1289          } else if (gr.kind == K_R1) {
 1290            const int t = t0 + ci;
 1291            if (lane < ntok) {
 1292              sd_row[s][lane] = tokl;
 1293              sd_f[s][lane] = fl;
 1294            }
 1295            __syncwarp();
 1296            if (lane == 0) {
 1297              desc[s] = make_int4(hdr, ei, t, ntok);
 1298              mbar_expect_tx(&full[s],
 1299                             W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
 1300              tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
 1301              tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
 1302            }
 1303            // weights are in flight; only the activation rows wait for the
 1304            // entry's ready (every copying lane acquires for itself)
 1305            if ((ei | q << 16) != last_ready) {
 1306              if (!pre_ready) {
 1307                while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
 1308              }
 1309              fence_proxy_async();
 1310              last_ready = ei | q << 16;
 1311            }
 1312            __syncwarp();
 1313            if (lane < ntok && lane < XROWS)
 1314              bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
 1315                       ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
 1316                       &full[s]);
 1317          } else {  // shared expert
 1318            if (lane == 0) {
 1319              desc[s] = make_int4(hdr, gr.x, c, 0);
 1320              mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
 1321              tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
 1322                     &full[s]);
 1323            }
 1324            if (gr.kind == K_S1 &&
 1325                last_ready != -2) {  // every copying lane acquires
 1326              while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
 1327              fence_proxy_async();
 1328            }
 1329            if (gr.kind == K_S1) last_ready = -2;
 1330            __syncwarp();
 1331            for (int t = lane; t < T; t += 32)
 1332              bulk_g2s(
 1333                  dst + W_BYTES + t * XS_BYTES,
 1334                  (gr.kind == K_S0 ? ws->x13b + static_cast<size_t>(t) * HIDDEN
 1335                                   : ws->x2s + static_cast<size_t>(t) * INTER) +
 1336                      c * CKS,
 1337                  XS_BYTES, &full[s]);
 1338          }
 1339        }
 1340        __syncwarp();
 1341        if (lane == 0) mbar_arrive(&sq_empty[k]);
 1342        if (++k == TD_SQ) {
 1343          k = 0;
 1344          kph ^= 1;
 1345        }
 1346      }
 1347      if (lane == 0) {  // end of work
 1348        const int s = it % STAGES;
 1349        if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
 1350        desc[s] = make_int4(K_END, 0, 0, 0);
 1351        mbar_arrive(&full[s]);
 1352      }
 1353      return;
 1354    }
 1355  
 1356    // consumer warps
 1357    const int g = lane / 4, tq = lane % 4;
 1358    const int nt_n = (T + 7) / 8;
 1359    float acc[4][4] = {};
 1360    float accs[2][NT_MAX][4] = {};
 1361    int hk = 0, hn = 0;  // handoff slot and count (same in every consumer)
 1362    const auto hand_off = [&](int4 h) {
 1363      // every warp waits for the slot's previous use to be taken before it
 1364      // arrives (no arrival may enter a phase whose predecessor nobody waited
 1365      // on); __syncwarp gathers the warp's reds and warp 0's info for lane 0's
 1366      // release
 1367      if (lane == 0) {
 1368        if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
 1369        if (warp == 0) hq_info[hk] = h;
 1370      }
 1371      __syncwarp();
 1372      if (lane == 0) mbar_arrive(&hq_full[hk]);
 1373      if (++hk == TD_HQ) hk = 0;
 1374      ++hn;
 1375    };
 1376    // per-warp smem offsets within a stage, computed once
 1377    const uint32_t ring_u = smem_u32(ring), full_u = smem_u32(full),
 1378                   empty_u = smem_u32(empty);
 1379    const uint32_t desc_u = smem_u32(desc), sdf_u = smem_u32(&sd_f[0][2 * tq]);
 1380    const uint32_t scr_u = smem_u32(scr_all[warp]),
 1381                   sdr0_u = smem_u32(&sd_row[0][0]);
 1382    static_assert(CONSUMER_WARPS == 8,
 1383                  "8 K slices (w13), 2 halves x 4 K slices (w2)");
 1384    const int h1 = warp / 4, k1 = warp % 4;
 1385    const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
 1386    const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
 1387    const uint32_t xo1 =
 1388        W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (k1 * 16 + tq) * 16;
 1389    int s = 0;
 1390    uint32_t ph = 0;
 1391    for (int it = 0;; ++it) {
 1392      mbar_wait_a(full_u + 8 * s, ph);
 1393      const uint4 du = lds_v4(desc_u + 16 * s);
 1394      const int4 d = make_int4(du.x, du.y, du.z, du.w);
 1395      const int kind = d.x & 15;
 1396      if (kind == K_END) {
 1397        hand_off(make_int4(0, 0, 0, 1));
 1398        break;
 1399      }
 1400      const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
 1401      const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
 1402      const uint32_t st_u = ring_u + s * STAGE_BYTES;
 1403      const uint32_t empty_s = empty_u + 8 * s;
 1404      // this lane's two tokens of a routed unit: destination rows and scales
 1405      const int ntok = d.w;
 1406      // flush metadata of stage s (plain shared loads the compiler may schedule
 1407      // into the math), read before the stage is released
 1408      const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
 1409        fs_[0] = sd_f[s_][2 * tq];
 1410        fs_[1] = sd_f[s_][2 * tq + 1];
 1411        if (r1) {
 1412          const unsigned char* xr =
 1413              ring + static_cast<size_t>(s_) * STAGE_BYTES + W_BYTES + S_BYTES +
 1414              ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
 1415          fs_[0] *= *reinterpret_cast<const float*>(xr);
 1416          fs_[1] *=
 1417              *reinterpret_cast<const float*>(xr + (XROWS > 1 ? XROW_STRIDE : 0));
 1418        }
 1419  #pragma unroll
 1420        for (int j = 0; j < MAX_TOK / 2; ++j)
 1421          rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
 1422      };
 1423      float fs[2];
 1424      int rows4[MAX_TOK / 2];
 1425  
 1426      if (kind == K_R0) {
 1427        // the group's nch chunks in a row: its later stages carry nothing new
 1428        // for the consumer (same entry, tile, tokens), so no descriptor reads
 1429        for (int c = 0;; ++c) {
 1430          const uint32_t su = ring_u + s * STAGE_BYTES;
 1431          consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
 1432          if (c == nch - 1) flush_meta(s, fs, rows4, false);
 1433          __syncwarp();
 1434          if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
 1435          if (c == nch - 1) break;
 1436          if (++s == STAGES) {
 1437            s = 0;
 1438            ph ^= 1;
 1439          }
 1440          mbar_wait_a(full_u + 8 * s, ph);
 1441        }
 1442        {
 1443          const int ei = d.y, t = d.z;
 1444          flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
 1445                     ntok, g, tq, lane);
 1446          hand_off(make_int4(q, ei, nch, 0));
 1447        }
 1448      } else if (kind == K_R1) {
 1449        consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
 1450        flush_meta(s, fs, rows4, true);
 1451        __syncwarp();
 1452        if (lane == 0) mbar_arrive_a(empty_s);
 1453        const int t = d.z;
 1454        flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
 1455                   g, tq, lane);
 1456      } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
 1457        const unsigned char* xa = st + W_BYTES;
 1458  #pragma unroll
 1459        for (int ks = 0; ks < CKS / 16; ++ks) {
 1460          uint32_t af[2][4];
 1461  #pragma unroll
 1462          for (int rb = 0; rb < 2; ++rb) {
 1463            const int row =
 1464                warp * 32 + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
 1465            const int ck = ks * 2 + (lane >> 4);
 1466            ldmatrix_x4(af[rb], st + row * 128 + ((ck ^ (row & 7)) << 4));
 1467          }
 1468  #pragma unroll
 1469          for (int nt = 0; nt < NT_MAX; ++nt) {
 1470            if (nt < nt_n) {
 1471              const uint2 bv = *reinterpret_cast<const uint2*>(
 1472                  xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 +
 1473                  (ks & 1) * 8);
 1474              mma_bf16(accs[0][nt], af[0], bv.x, bv.y);
 1475              mma_bf16(accs[1][nt], af[1], bv.x, bv.y);
 1476            }
 1477          }
 1478        }
 1479        __syncwarp();
 1480        if (lane == 0) mbar_arrive_a(empty_s);
 1481        if (last) {
 1482          const int t = d.y;
 1483  #pragma unroll
 1484          for (int rb = 0; rb < 2; ++rb)
 1485  #pragma unroll
 1486            for (int nt = 0; nt < NT_MAX; ++nt)
 1487  #pragma unroll
 1488              for (int i = 0; i < 4; ++i) {
 1489                const int tok = nt * 8 + 2 * tq + (i & 1);
 1490                const int row = t * RS + warp * 32 + rb * 16 + g + (i >> 1) * 8;
 1491                if (nt < nt_n && tok < T) {
 1492                  if (kind == K_S0)
 1493                    atomicAdd(
 1494                        &ws->y13s[static_cast<size_t>(tok) * 2 * INTER + row],
 1495                        accs[rb][nt][i]);
 1496                  else
 1497                    atomicAdd(&ws->y[static_cast<size_t>(tok) * HIDDEN + row],
 1498                              p.shared_scale * accs[rb][nt][i]);
 1499                }
 1500                accs[rb][nt][i] = 0.f;
 1501              }
 1502          if (kind == K_S0) {
 1503            __threadfence();  // every thread's own y13s atomics, before the count
 1504            consumer_sync();
 1505            if (threadIdx.x == 0) {
 1506              __threadfence();
 1507              s_last = atomicAdd(&ws->done_s, nch) + nch == UNITS_S0;
 1508            }
 1509            consumer_sync();
 1510            if (s_last) {
 1511              __threadfence();
 1512              activate_shared(ws, T, warp, lane);
 1513              fence_proxy_async();
 1514              __threadfence();
 1515              consumer_sync();
 1516              if (threadIdx.x == 0) st_release(&ws->ready_s, epoch);
 1517            }
 1518          }
 1519        }
 1520      }
 1521      if (++s == STAGES) {
 1522        s = 0;
 1523        ph ^= 1;
 1524      }
 1525    }
 1526  }
 1527  
```
### host entry (1543-1668)
```cuda
 1543  // ---------------------------------------------------------------- host entry
 1544  // Plain C ABI (no torch headers: the build is the kernel alone), called from
 1545  // Python through ctypes with raw device pointers and the current stream.
 1546  #define TD_REQUIRE(c, msg)                              \
 1547    do {                                                  \
 1548      if (!(c)) {                                         \
 1549        std::fprintf(stderr, "tiered_decode: %s\n", msg); \
 1550        return -1;                                        \
 1551      }                                                   \
 1552    } while (0)
 1553  
 1554  // [n2][n1][n0] contiguous elements of elem bytes; box {b0, b1, 1}
 1555  static int make_map3(CUtensorMap* map, const void* p, uint64_t n0, uint64_t n1,
 1556                       uint64_t n2, int elem, CUtensorMapDataType type,
 1557                       uint32_t b0, uint32_t b1) {
 1558    const cuuint64_t dims[3] = {n0, n1, n2};
 1559    const cuuint64_t strides[2] = {n0 * elem, n0 * n1 * elem};
 1560    const cuuint32_t box[3] = {b0, b1, 1}, unit[3] = {1, 1, 1};
 1561    return cuTensorMapEncodeTiled(
 1562               map, type, 3, const_cast<void*>(p), dims, strides, box, unit,
 1563               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
 1564               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
 1565               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
 1566               ? 0
 1567               : -1;
 1568  }
 1569  // [rows][cols] bf16 with a {b0, b1} box, 128 B swizzled
 1570  static int make_map_2d_sw128(CUtensorMap* map, const void* p, uint64_t cols,
 1571                               uint64_t rows, uint32_t b0, uint32_t b1) {
 1572    const cuuint64_t dims[2] = {cols, rows};
 1573    const cuuint64_t strides[1] = {cols * 2};
 1574    const cuuint32_t box[2] = {b0, b1}, unit[2] = {1, 1};
 1575    return cuTensorMapEncodeTiled(
 1576               map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p),
 1577               dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE,
 1578               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
 1579               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
 1580               ? 0
 1581               : -1;
 1582  }
 1583  static int fill_tier(Tier& tr, const void* w13, const void* s13, const void* w2,
 1584                       const void* s2, int e) {
 1585    std::memset(&tr, 0, sizeof(tr));
 1586    if (e == 0) return 0;
 1587    int r = make_map3(&tr.w[0], w13, 2 * INTER * 2, HIDDEN / 16, e, 4,
 1588                      CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R0, KT0);
 1589    r |= make_map3(&tr.w[1], w2, HIDDEN * 2, INTER / 16, e, 4,
 1590                   CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R1, KT1);
 1591    r |= make_map3(&tr.s[0], s13, 2 * INTER, HIDDEN / 32, e, 2,
 1592                   CU_TENSOR_MAP_DATA_TYPE_UINT16, R0, G0);
 1593    r |= make_map3(&tr.s[1], s2, HIDDEN, INTER / 32, e, 2,
 1594                   CU_TENSOR_MAP_DATA_TYPE_UINT16, R1, G1);
 1595    return r;
 1596  }
 1597  
 1598  }  // namespace tiered_decode
 1599  
 1600  using namespace tiered_decode;
 1601  
 1602  extern "C" long long td_workspace_bytes() { return sizeof(Workspace); }
 1603  
 1604  extern "C" int td_forward(void* out, const void* x, const int* ids,
 1605                            const float* wt, const int* hot_map,
 1606                            const int* cold_map, int T, const void* hw13,
 1607                            const void* hs13, const void* hw2, const void* hs2,
 1608                            int hot_size, const void* cw13, const void* cs13,
 1609                            const void* cw2, const void* cs2, int cold_size,
 1610                            void* workspace, int pdl_launch, const bool* padding,
 1611                            const void* sw13, const void* sw2, float shared_scale,
 1612                            float routed_scale, int num_experts,
 1613                            void* stream_ptr) {
 1614    TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..MAX_TOKENS tokens");
 1615    TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
 1616    TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS, "1..512 experts");
 1617    Params p{};
 1618    TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
 1619               "hot tier maps");
 1620    TD_REQUIRE(fill_tier(p.tier[1], cw13, cs13, cw2, cs2, cold_size) == 0,
 1621               "cold tier maps");
 1622    p.ws = reinterpret_cast<Workspace*>(workspace);
 1623    p.has_shared = sw13 != nullptr;
 1624    p.shared_scale = shared_scale;
 1625    if (p.has_shared) {
 1626      TD_REQUIRE(
 1627          make_map_2d_sw128(&p.sw[0], sw13, HIDDEN, 2 * INTER, CKS, RS) == 0,
 1628          "sw13 map");
 1629      TD_REQUIRE(make_map_2d_sw128(&p.sw[1], sw2, INTER, HIDDEN, CKS, RS) == 0,
 1630                 "sw2 map");
 1631    }
 1632    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
 1633    static bool attrs = false;
 1634    if (!attrs) {
 1635      cudaFuncSetAttribute(
 1636          layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
 1637      attrs = true;
 1638    }
 1639    cudaLaunchAttribute pdl[1];
 1640    pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 1641    pdl[0].val.programmaticStreamSerializationAllowed = 1;
 1642    auto config = [&](dim3 grid, dim3 block, size_t smem) {
 1643      cudaLaunchConfig_t c = {};
 1644      c.gridDim = grid;
 1645      c.blockDim = block;
 1646      c.dynamicSmemBytes = smem;
 1647      c.stream = stream;
 1648      c.attrs = pdl;
 1649      c.numAttrs = pdl_launch ? 1 : 0;
 1650      return c;
 1651    };
 1652    Placement pl{};
 1653    const size_t route_smem =
 1654        2 * static_cast<size_t>(hot_size + cold_size) * sizeof(int);
 1655    cudaLaunchConfig_t c = config(dim3(T + 1), dim3(PREP_THREADS), route_smem);
 1656    if (cudaLaunchKernelEx(&c, route_prep_kernel<int>, p.ws,
 1657                           reinterpret_cast<const __nv_bfloat16*>(x), ids,
 1658                           padding, wt, hot_map, cold_map, pl, T, hot_size,
 1659                           cold_size, num_experts, routed_scale) != cudaSuccess)
 1660      return -2;
 1661    c = config(dim3(GRID), dim3(THREADS), SMEM_BYTES);
 1662    if (cudaLaunchKernelEx(&c, layer_kernel, p) != cudaSuccess) return -3;
 1663    c = config(dim3(HIDDEN / 1024, T), dim3(256), 0);
 1664    if (cudaLaunchKernelEx(&c, finalize_kernel, p.ws,
 1665                           reinterpret_cast<__nv_bfloat16*>(out)) != cudaSuccess)
 1666      return -4;
 1667    return 0;
 1668  }
```

## SentryWard facts (from its source)
- lowering.cpp:1486: `if (s.atomic_add && (s.category == "memory" || !s.modeled)) throw SemanticError("atomic_add_memory_semantics_unsupported", ...)`
- semantics.cpp: `wgmma` and generic `cp` async marked modeled=false; `fence`/`membar` classified as category fence (no ordering effect in the IR); `griddepcontrol` -> external_control.
- Event kinds (event_ir/event.h): ImmediateMemoryIssue/Exec, AsyncMemoryIssue/Exec, GenericAsyncIssue/Effect/ReadsComplete/Complete, OperationIssue/Complete, AwaitIssue/Done, MBarrierInit/Update/CompleteTx/AddArrivals/PhaseComplete, WaitState/WaitParity Issue/Done, Collective Issue/Done/Complete, NamedBarrierArrive/Return. No RMW, no release/acquire, no fence events.
- Executor: per-lane coroutine tasks, data flows between lanes only via a CTA shared-memory constant runtime (SharedExecution); multiple grid CTAs share one event graph but global memory carries no values between them.
- docs/assumptions.md: "Immediate visibility remains a fixed premise of the current model"; "Recognized polling waits keep their existing wait normalization; unmodeled polling remains unsupported."
