# Cold-expert H2D prefetch for prefill

Hide the cold tier's Grace->HBM traffic behind layer compute during prefill, so
cold Marlin reads weights from HBM at hot's rate instead of demand-reading them
across C2C.

**Target: -373 ms per 8192-token chunk (19.0%)**, ~-4.5 s on a 23.2 s TTFT at
96K. That is an upper bound, not a forecast.

## Evidence

All from `2026-09-04-mtp3-profile` (job 1665068), rank 0, 6 chunks.

* Hot and cold run the **same kernel, same grid `(264,1,1)`, same operand
  shapes, same M-tiling**. The only variable is where the weights live, and
  cold is **2.8x slower per expert** (156 vs 55 us for w13).
* Cold is clock-insensitive on the slow rank (+1.2%, where a compute-bound GEMM
  is +8.5%), so it is not SM-bound.
* Cold streams 52.1 GB per chunk at 91 GB/s logical. A bulk contiguous DMA at
  450 GB/s needs 115.8 ms -- a **6.1% duty cycle**.
* Measured window from "layer L's cold buffer is consumed" to "layer L+1's cold
  Marlin starts": min **12.96 ms**, median 16.11, against a 1.65 ms median
  need. **0 of 444 transitions fail** at 450 GB/s or at the 373 GB/s measured
  rate; worst case 7.04x. The window excludes cold Marlin by construction, so
  it does not shrink as cold speeds up.

## Design

A per-model coordinator owns two HBM staging slots and a copy stream. Layers
execute in order and `apply_tiered_moe` runs once per layer, so the coordinator
is driven entirely by those calls -- no scheduler hook.

For layer L:

1. Compute stream waits the copy event for slot `L % 2`.
2. Hot tier runs unchanged.
3. Cold tier runs against slot `L % 2`'s component views.
4. Record `consumed[L % 2]` on the compute stream.
5. Copy stream waits `consumed[(L+1) % 2]`, issues the `cudaMemcpyAsync` for
   layer L+1's cold bytes into that slot, and records `copied[(L+1) % 2]`.

**No zeroing.** The slot is fully overwritten by the incoming copy; clearing it
first would cost 0.17-0.21 ms/layer of HBM write bandwidth for nothing.

**One memcpy per layer, and the views must be rebuilt -- not sliced.**
`build_expert_component_views` lays a tier out **component-major**: all of w13
for every expert, then all of w2, then the scales. A layer's Grace buffer and a
slot prefix of the same expert count therefore have byte-identical layouts, so
one contiguous `cudaMemcpyAsync` of the whole layer buffer into the slot's front
is correct. But a slot sized for 41 experts holds w2 at a *different* offset
than a 33-expert layer does, so **slicing a max-sized view is silently wrong** --
it would scatter w2 and the scales to the wrong offsets, and nothing but the
bitwise gate would catch it. Views are therefore built per `(layer, slot)` at
init against `slot.narrow(0, 0, n * expert_bytes)`, 2 x 75 view dicts, none of
it on the hot path.

The two schemes have different windows, and the measured numbers belong to the
stricter one:

| scheme | copy for L+1 may start at | window | slots |
| --- | --- | ---: | ---: |
| single buffer | end of cold L | **12.96 ms min (measured)** | 0.81 GiB |
| double buffer | end of cold L-1 | ~that plus one layer | 1.62 GiB |

Both fit with margin. Start with double, which is the design above; single is
the fallback if HBM is tight.

### Sizing -- resolved, not assumed

Two layers per chunk hold all 64 owned experts in a single Marlin call, which
would break the sizing if they were cold. They are **hot**: `all64_tier_probe.py`
measures them at 250/228 GB/s against a hot median of 268/243 and a cold median
of 93/92, and two further all-64 calls run in 22 us for 864 MiB -- degenerate
launches with no routed rows. **No all-cold-64 layer exists.**

| | |
| --- | ---: |
| largest cold layer | 41 experts, 830 MiB |
| two slots | **1.62 GiB** |

Slots are sized at init from the **placement**, as `max(cold_expert_count)` over
layers, not from a runtime observation -- and a layer whose cold count exceeds
the slot falls back to the direct Grace path rather than failing.

Cold storage is `cold_expert_ids + replica_expert_ids`, so a deployment with
replicas would have a cold buffer larger than the set prefill actually routes
to. Here it does not: `prepare_replica_routing` returns early above
`tiered_replica_max_tokens`, so **replicas are a decode-only mechanism**, and
hot + cold sums to exactly the 64 experts each rank owns in all 3600 validated
layer pairs -- replica_expert_ids is empty. If replicas are ever enabled, the
slot must be sized to the *buffer* and the prefetch would copy bytes prefill
never reads; size from the buffer regardless.

### HBM must come from the planner, not the runtime margin

The observed margin is 1.95 GiB (10.33 free, 8.38 minimum), and taking 1.62 out
of it leaves 0.33 against a 1.0 GiB tolerance -- `verify_observed_hbm_reserve`
would reject it. This is the trap that cost three launches in Phase 53.

So the slots must be **budgeted**, exactly as `draft_cache` was: add
`"cold_staging"` to `fixed_hbm_allocations` and `_DERIVED_ALLOCATION_NAMES` in
`tiered_moe_physical.py`, so the planner demotes hot experts to make room.
Expected cost **~82 experts, 2325 -> ~2243 hot** -- which costs nothing once
cold runs at hot's rate.

### Where it hooks

| file | change |
| --- | --- |
| `tiered_moe_storage.py` | staging-slot allocation, per-layer component views |
| `tiered_moe_physical.py` | `cold_staging` in the planner's fixed HBM allocations |
| `tiered_moe_execution.py` | coordinator; swap the cold tier's `w13`/`w2` for staged views |
| `tiered_moe_runtime.py` | coordinator lifetime alongside existing runtime state |

`apply_tiered` in `modular_kernel.py` already takes each tier's weight tensors
as arguments, so the swap happens at the call site and the kernel path is
untouched.

### Gating

New knob `VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS` (default 1024), a property
of the **batch**, not a global, so decode steps in the same process take the
existing path. It must not touch `overlap_max_tokens`, which defaults to 4 and
governs the unrelated decode tier-overlap stream. Prefill is eager here
(`graph-replayed 0%` in the capture), so there is no CUDA-graph capture
interaction -- but the gate being per-batch is what keeps that true.

## Why prefill only

Structural, not bandwidth. At M=8192 every owned expert fires, so "which cold
experts does layer L+1 need" is answerable before layer L runs: all of them. At
decode's M=4 roughly one cold expert per rank per layer fires, and which one is
unknown until that layer's router runs. Decode's cold path is also already at
the C2C roofline (379 GB/s). **Hot residency stays.**

## Correctness

The arithmetic does not change -- same kernel, same weight bytes, different
address -- so identical output is the expected result, and the gate is there to
catch the two ways the plumbing can break:

* **Staleness**: cold L+1 reads its slot before the copy landed (missing wait,
  or the event recorded on the wrong stream).
* **Clobber**: the copy for L+1 lands in a slot that cold L-1 is still reading.

Both corrupt weights, and both show up as changed logits. Gate: the same prompt
at temperature 0 produces **bitwise-identical** output with the flag on and off.
The unit test asserts the event ordering explicitly -- `copied` recorded on the
copy stream and waited by the compute stream, `consumed` recorded on the compute
stream and waited by the copy stream -- not merely that slots rotate.

## Phases

1. **Microbenchmark first, in `benchmarks/`.** Does a Grace-pinned -> HBM
   `cudaMemcpyAsync` sustain >=400 GB/s on a copy stream *while* Marlin runs on
   the compute stream? Measure the copy rate and the Marlin slowdown. This is
   the cheapest way to kill the idea, because the design assumes the DMA is both
   fast and non-disruptive.

   Measure **that path specifically**, in that direction. The 373 GB/s this
   project carries was measured for SM-issued loads over UVA
   (`benchmarks/pageable_grace_bandwidth.py`), and a copy-engine H2D from pinned
   Grace is a different mechanism that may be faster or slower. Also confirm
   each rank's cold buffer is pinned on its **own** socket: C2C is per-superchip,
   so four ranks use four independent links and do not contend -- but only if
   each buffer is NUMA-local. If it is not, the copy crosses the CPU-CPU
   interconnect and 450 GB/s is the wrong ceiling. This project has had that
   wrong before (`numa-bind-uva`).
2. Slot allocation and the copy path, behind the flag, cold tier still reading
   Grace -- verifies plumbing without changing results.
3. Switch the cold tier to the staged views; bitwise gate.
4. Planner accounting for `cold_staging`.
5. A/B on one allocation, both arms, same node and prompt.

## Measurement and kill criteria

Compare cold Marlin ms/chunk, chunk wall, and TTFT at 96K. Confirm from the
trace that the H2D memcpy is **concurrent with** compute kernels rather than
serialized between them -- that is the thing being built, and it is visible.

**Kill:** if **cold Marlin does not fall from 583 ms/chunk to below 383 ms**,
measured the same way through the same trace analysis, the premise -- that
residency is what makes cold slow -- is wrong. Stop rather than tune.

**Chunk boundary.** The first MoE layer of a chunk has no predecessor to
prefetch behind, so it eats one exposed copy of ~1.6 ms in a 1965 ms chunk.
Accept it rather than special-casing the dense layers 0-2.

## Risks

1. **The 373 ms is an upper bound.** It assumes cold-in-HBM reaches hot's
   85 us/expert. Cold experts are cold because they take fewer tokens, so
   Marlin may get worse M-utilisation from them. Against that: hot itself is at
   neither the HBM roof (250 GB/s of 4000) nor the SM clock, so hot's rate is
   set by a residency-**independent** kernel limit that cold should converge to.
   Row counts are not in the trace (shapes, not values); measure with a
   `topk_ids` histogram over one prefill, or reuse
   `benchmarks/capture_routing_trace.py`, before trusting the upper bound.
2. **Copy/compute contention.** The DMA writes at 450 GB/s into the same HBM
   the compute reads -- 11% of HBM, but not free. Phase 1 measures it.
3. **The reserve check**, addressed by planner accounting above; it has blocked
   three launches before.
4. **NUMA locality of the cold buffers.** C2C is per-superchip, so each rank
   prefetches over its own link and the four do not contend -- provided each
   rank's cold buffer is pinned on its own socket. Confirm in phase 1; if it is
   not local, the copy crosses the CPU-CPU interconnect and the 450 GB/s
   ceiling does not apply.
