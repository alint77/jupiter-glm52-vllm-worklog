# Extending SentryWard to the sliced decode MoE kernel

## Status after Astra's review (`../astra/extension-review.md`)

The original draft follows unchanged; this section supersedes it where they differ.

**Verdict:** GRID=1 is a good first milestone. S2–S4 as drafted could accept
impossible executions or miss ordering obligations. No definite
synchronization bug was found in `layer_kernel`.

**Corrections:**

1. **S2 (pinned modification order + fixpoint + hb check) is not enough.** A
   candidate execution also needs reads-from choices, event existence,
   returned values, control-flow consistency and SC-fence order. For example,
   two acquire probes reading `epoch` then `epoch-1` with one publication
   breaks no RMW order, yet is forbidden.

   Replace S2 with an **executable replay**:
   - pick among enabled actions;
   - run RMWs against shared global state;
   - record order and reads-from as the witness.

   The interpreter's scheduling order must never become an hb edge between
   unrelated threads. Report `feasible-and-checked`, `infeasible-candidate` or
   `unknown`. The counters have constant increments per location (`done13` +=
   R0CH, `done_s` += SCH0), so no variable-increment fixpoint is needed.
2. **Use PTX release/acquire patterns and observation order,** not C++ release
   sequences. Specifics:
   - a relaxed RMW in the middle of a chain carries earlier publications but
     doesn't publish its own payload;
   - `red` never acquires;
   - `membar.gl` = `fence.sc.gpu` needs **full Fence-SC order**, not just
     acquire+release.
3. **`done_s` (lines 1503–1512) is ordered only by the SC-fence argument.**
   Only thread 0 reads `done_s`, and the `bar.sync` at 1509 precedes its
   acquiring fence at 1511. Write it as a dedicated litmus test first.
   Production hardening option: an acq_rel `done_s` RMW before the `bar.sync`.
   Don't make that change only in the verification build.
4. **S3 must separate observing a value from acquiring its payload.**
   - A relaxed store of `epoch` still ends the spin; what's lost is
     publication, which is not `await_no_writer`.
   - The probe outcome needs a legal reads-from source either way. "All probes
     false" can be infeasible.
   - A cycle created by an assumed wait edge proves the pinned choice is
     impossible, not that the kernel deadlocks.
   - Termination stays an explicit progress/fairness assumption.
5. **S4:**
   - The consumer-side `fence.proxy.async.global` (lines 1309, 1327) is the
     right placement; the remote writer's fence (1513) is no substitute.
   - The **blanket WAR exemption is not justified**: stage reuse after
     generic reads needs synchronization plus the proxy fence, which
     line 1268/1275 supplies. Check it rather than exempt it.
   - Distinguish issue from completion. A bulk `complete_tx` publishes the
     copy, not the issuing thread's earlier operations.
6. **S0/S1:**
   - INTER stays 512.
   - With GRID ≤ 3 and both tiers populated, `TD_COLD_CTAS=16` puts every CTA
     on the cold tier, so use `-DTD_COLD_CTAS=1` for mixed splits.
   - Prefer synthetic, contract-checked images (layout, extents, route
     bounds/uniqueness, `ntok`/counts, queue membership) over byte dumps from a
     differently sized build.
   - GRID=1 makes the **control counters** single-writer. The y13/y reds still
     come from many warps.
7. **Race freedom is not the whole property.** Add protocol assertions:
   - each group processed exactly once;
   - each completion after its group's reds;
   - totals match;
   - one publication, after full accumulation;
   - activation before zeroing;
   - x2 unchanged while it is read;
   - every claimed group completes.

   Also prove w13 progress doesn't depend on a blocked w2 producer, in a small
   scheduler model including stealing. Model vector `red` per element.

**Revised plan (Astra's effort estimates are for someone who knows
SentryWard):**

| Step | Deliverable | Effort |
|---|---|---|
| 1 | Hand-extracted litmus tests (Dartagnan PTX target, model audited per feature): routed `done13` counter, shared `done_s` SC-fence collection, scheduler acquire relay through `sq`, proxy placement and stage reuse | 2–5 days |
| 2 | S0 (done), validated synthetic S1 fixtures, S5 interpreter support | 3–7 days |
| 3 | GRID=1: executable control-counter semantics, faithful flag reads, proxy checks, protocol assertions | 5–10 days |
| 4 | Group-level scheduler model (ownership, stealing, outstanding work, termination) + GRID=2–3 replay cases | 3–7 days |
| 5 | Full pinned/symbolic PTX execution consistency, only if needed | 3–6+ weeks |

First milestone, stated as: "for this input image and witnessed control
execution, the modeled accesses satisfy the checked ordering and protocol
obligations, under the listed initialization, proxy and progress assumptions."

---

## Original draft

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

Done: `TD_HIDDEN` and `TD_GRID` macros. With the defaults, the PTX is
byte-identical to HEAD.

- **INTER is fixed at 512.** These asserts pin it: `W_BYTES == R1*INTER/2`
  with R0=R1=128 and CK0=512, and `XROW_BYTES0 == XROW_BYTES1`.
- **HIDDEN=3072 keeps every assert:**
  - CHUNKS0 = 6 > STAGES;
  - CHUNKS_S0 = 48 is divisible by SCH0 = 12;
  - TILES1 = 24 is divisible by GR1.
- **Units:**
  - per routed entry: 48 w13 + 24 w2 = 72 (vs 144 at 6144);
  - shared expert: 192 w13 + 96 w2.
- **Launch:** `-DTD_HIDDEN=3072 -DTD_GRID=1`. With GRID=1, CTA 0 takes the cold
  tier and steals the hot one.

Two routings: one routed expert per tier and no shared expert; and one routed
expert plus the shared expert.

Opcodes in that build's `layer_kernel` (10.7K PTX lines) that SentryWard
lacks:

| Opcode | Count |
|---|---|
| `atom.global.add.f32` | 128 |
| `red.global.v4.f32.add` | 8 |
| `atom.global.add.u32` (claims, `done_s`) | 5 |
| `atom.acq_rel.gpu.global.add.s32` | 1 |
| `ld.acquire.gpu` | 5 |
| `st.release.gpu` | 2 |
| `membar.gl` | 4 |
| `fence.proxy.async.global` | 3 |
| `fence.proxy.async.shared::cta` | 1 |
| `nanosleep` | 2 |
| `griddepcontrol` | 2 |
| `lop3` | 256 |

`ld.global.cg.f32`: 160, the y13 reads in `activate_routes`.

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
