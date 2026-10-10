# Astra review 13: ship review of the sliced MoE decode kernel (kernel, adversarial tests, results)

Do NOT run any commands (your sandbox cannot run here). Everything you need is inlined below.

## What is being shipped

The full source of `vllm/model_executor/layers/fused_moe/tiered_decode/sliced_decode.cu` is at the end.
It replaces the served kernel (v31 lineage) with v53 + TD_NOAHEAD, cleaned for shipping:
- It was generated mechanically: v53 with -DTD_NOAHEAD, then TD_* `#if` blocks resolved, trace hooks
  stripped, the unused ready-list (TD_RLIST) storage removed, and the disabled inline w13 handoff
  (`bar.sync 2`) removed.
- Verification: cuobjdump SASS of layer_kernel and finalize_kernel is byte-identical to
  `td_v53.cu -DTD_NOAHEAD`; ptxas reports 168 regs and 0 spills.
- route_prep_kernel differs only by the two dropped `rl_tail` zero stores, plus the resulting branch
  offsets.
- Tunables stay overridable through `#ifndef` (TD_STAGES 4, TD_HQ 4, TD_ACTK 4, TD_SQ 2,
  TD_COLD_CTAS 16, TD_GR1 2, TD_GR1_T 8, TD_R0S 1, TD_FIN_FENCE).

What is already reviewed (your reviews 10-12, applied as you wrote them):
- The scheduler -> producer smem FIFO (sq_full/sq_empty, TD_SQ=2), including the ready probe carried
  through the FIFO.
- The finisher warp with the hq_full/hq_empty handoff. Every consumer warp's lane 0 waits hq_empty
  before reuse; then __syncwarp; then a lane-0 arrive; hq_full counts CONSUMER_WARPS. The producer
  checks K_END before reading the Group.
- Review 12 defaults:
  - y13 is zeroed after the ready release (float4 stores).
  - fence.acq_rel.gpu.
  - No GPU fence before the release __syncwarp (ONEREL).
  - Lane 0's count is one atom.acq_rel.gpu.
  - No writer-side proxy fence.
  - `yr[j]` loads sit inside the route guard.
  - ACTK=4 route interleave.

New since review 12, and in the shipped code:
- **TD_NOAHEAD.** The scheduler normally claims the next group right after publishing one (claim-ahead).
  It no longer does that when the group it just published is a routed or shared w13 group with
  several chunks ("heavy": `gi < len[q] && group_at(...).kind != K_R1`). After publishing such a
  group it waits `sq_empty[k]` with parity `(n / TD_SQ) & 1`, i.e. until the producer has taken that
  slot, and only then claims.
- Motivation: at M=8 a CTA's claim-ahead hoarded a 12-chunk w13 group that idle CTAs could have
  taken, which delayed an entry's ready and every w2 unit waiting on it.
- Look for this in the scheduler warp: `const bool heavy = ...`, then
  `if (lane == 0 && !heavy) gn = atomicAdd(...)`, and after `mbar_arrive(&sq_full[k])`,
  `if (heavy && lane == 0) { mbar_wait(&sq_empty[k], (n / TD_SQ) & 1); gn = atomicAdd(...); }`.
- Please check the parity: n is the count of slots published before this one, and the other waits on
  sq_empty use the same scheme.
- Please also check that this cannot deadlock against the R0-before-dependent-R1 claim-order
  invariant from review 11. The wait depends only on the producer accepting the slot, and the
  producer can be blocked by an R1 ready spin only for groups it already holds.

vLLM wrapper change (sliced.py): td_forward gained `int num_experts` before the stream argument,
and the wrapper passes hot_map.numel(). It raises if `hot_map.numel() != cold_map.numel()`.
route_prep stages both maps in smem, `maps[2 * MAX_EXPERTS]`, for i < num_experts. It then reads
maps[e] for e >= 0. In the placement-less path there is no bound check against num_experts.
- Question: should route_prep map ids >= num_experts to -1 (cost: one compare per route)?
- The old served kernel read hot_map[e] from global without a bound check.
- Router ids are always < E in vLLM.

## Adversarial check (adv_check.py, inlined below)

Fixture: 52 checkpoint experts (40 hot, 12 cold in Grace memory), all four 512-wide slices.
Reference: fp32 dequantized math (kdev.reference), with the shared expert scale 0.4 and the routed
scale 2.5.

Cases, at every M in {1,2,3,5,7,8,9,13,16,17,24,31,32}:
- Routing kinds:
  - sparse
  - same (every token on the same 8 experts)
  - heavy (one expert in every token)
  - cold-only
  - hot-only
  - masked (random -1, token 0 fully masked)
  - shared_only (all -1)
  - one_cold
  - one_hot
- Activation scales: normal 0.3 / large 30 / tiny 3e-4; even rows zeroed; and no-shared.

How the cases run:
- 4 slices per case, in random interleaved order across M and kind.
- The workspace is never reset.
- Outputs are NaN-prefilled.
- Pass if the error is at most 5e-3 relative to max |ref|, with no NaN, and exactly zero when ref is
  all zero.
- Then a CUDA graph of 6 different calls (M 32/1/8/17/32/5, mixed kinds) is replayed 50 x reps
  times as a PDL chain, NaN-refilled before every replay.

Results (each run: 2928 checked calls, 0 fails, 0 graph fails, worst rel err 0.00395, seed 0,
reps 3):
- td_v48
- td_v53
- td_v53:TD_NOAHEAD (the shipped code)
- td_v53:TD_NOAHEAD+TD_RLIST
- td_v53:TD_RLIST

The cleaned shipped source (td_v54) is running now with 4 more seeds at reps 5; vLLM's own
tests/kernels/moe/test_tiered_decode_sliced.py is running on the shipped file. I will report both
when you are done. The worst error, 0.00395, comes from bf16 output rounding plus fp32 summation
order; the M=8 bench check (separate script) has worst 0.00344.

## Performance results

Kernel bench (us, best of graph replays of 20 distinct routings, H/C = hot/cold experts touched;
roof = HBM+C2C floor / time):

M=8:
| variant | 30/2 | 38/0 | 38/4 | 50/4 |
|---|--:|--:|--:|--:|
| v48 | 77.8 | 85.8 | 86.7 | 106.1 |
| v53 NOAHEAD (ship) | 73.0 | 80.1 | 88.5 | 109.4 |
| v53 NOAHEAD+HOT | 71.7 | 80.2 | 88.2 | 108.7 |
| v53 RLIST | 75.1 | 82.3 | 87.1 | 108.4 |

M=32:
| variant | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|
| v48 | 143.9 | 207.6 | 231.4 |
| NOAHEAD | 140.8 | 198.5 | 230.6 |
| NOAHEAD+HOT | 144.2 | 197.1 | 234.2 |
| RLIST | 140.5 | 197.5 | 231.9 |

Replay against production EP, from the same node's grid benches. Method:
- Fit us = a + b·hot + c·cold per (variant, M).
- Replay 2,000 held-out agentic decode steps × 75 layers, using the production expert profile.
- For EP: replicas plus the deployed balancer, with the step time taken from the slowest GPU per
  layer.
- For TP-sliced: node-wide counts on the slice kernel.

MoE ms per step:
| | EP slowest GPU | EP mean GPU | slice-v53 | **NOAHEAD (ship)** | NOAHEAD+HOT |
|---|--:|--:|--:|--:|--:|
| M=8 | 8.75 | 7.53 | 7.06 | **7.01** | 7.30 |
| M=32 | 22.43 | 20.40 | 19.26 | **18.89** | 19.48 |

Fits (us = a + b hot + c cold):
| | a | b | c |
|---|--:|--:|--:|
| EP M=8 | 34.4 | 4.46 | 26.33 |
| ship M=8 | 30.5 | 1.27 | 4.14 |
| EP M=32 | 53.3 | 4.76 | 28.31 |
| ship M=32 | 34.6 | 1.34 | 5.73 |

The max fit residual is 22-35 us for the slice kernel and 38-66 us for EP.

## Questions

1. **Correctness of the shipped kernel end to end.** Prioritize the new NOAHEAD wait (parity, and no
   deadlock with stealing and K_END) and anything the mechanical cleanup could have broken. Ignore
   things you approved in reviews 10-12 unless the full source shows they are not as described.
2. **The num_experts bound** (above): guard it, or leave it as the caller's contract?
3. **Adversarial test.** What important cases does it miss? I can think of:
   - T=32 with all 256 routes on one cold expert.
   - Hot/cold slot maps that are non-identity permutations.
   - Experts present in neither tier (both maps -1 for an id that is routed).
   - Several back-to-back calls with PDL where the previous call's ready epochs alias.
   Which of these, or others, are worth adding before shipping? Is 5e-3 relative-to-max too loose to
   catch a missed route? One route of 8 contributes about 1/8 of the output.
4. **Results.** Is the replay comparison sound enough to call the kernel faster than EP? Is NOAHEAD
   over v53 (−0.05 ms at M=8, −0.37 ms at M=32) a real effect, given the fit residuals?
5. **Ship / no-ship**, and anything to fix first.

## adv_check.py
```python
"""Adversarial correctness check of a layer-kernel variant: the four 512
slices of 52 checkpoint experts (40 hot, 12 cold) against fp32, over routings
the round-1 check never produces: sparse top-8 over many experts, every token
on the same experts (entries split at 8 tokens), one heavy-hitter expert,
cold-only and hot-only, -1 padded routes (including fully masked tokens),
shared-only, large / tiny / zero activations, odd token counts, workspace
reused across M and routing changes, and a CUDA graph replaying several
different calls back to back (PDL chain).
  adv_check.py --v td_v53[:FLAGS] [--reps N] [--seed S]"""
import argparse
import json
import sys

import torch

import kdev
from kdev import HIDDEN, TOPK

N_HOT, N_COLD = 40, 12
E = N_HOT + N_COLD
SSCALE = 0.4


def fixture(dev):
    import bench_slice as BS

    T = BS.T
    gen = torch.Generator().manual_seed(1)
    ck = T._int4_experts(E, gen)
    slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier):
        hot = {k: v[:N_HOT].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[N_HOT:].contiguous() for k, v in tier.items()}, dev)
        return hot, cold

    sg = torch.Generator().manual_seed(7)
    sw13 = (torch.randn((4096, HIDDEN), generator=sg) * 0.02).to(torch.bfloat16)
    sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)
    rows = lambda r: torch.cat([torch.arange(512 * r, 512 * r + 512),  # noqa: E731
                                2048 + torch.arange(512 * r, 512 * r + 512)])
    return {"tiers": [split(sl) for sl in slices],
            "w13": T._int4_dequant(ck[0], ck[2]).to(dev),
            "w2": T._int4_dequant(ck[1], ck[3]).to(dev),
            "sw13": sw13.to(dev), "sw2": sw2.to(dev),
            "sslices": [(sw13[rows(r)].contiguous().to(dev),
                         sw2[:, 512 * r:512 * r + 512].contiguous().to(dev))
                        for r in range(4)]}


def routing(kind, m, g):
    def pick(pool, k):
        return torch.stack([pool[torch.randperm(len(pool), generator=g)[:k]]
                            for _ in range(m)])

    allx = torch.arange(E)
    if kind == "sparse":
        ids = pick(allx, TOPK)
    elif kind == "same":  # every token on the same 8 experts
        ids = pick(allx, TOPK)[:1].expand(m, TOPK).clone()
    elif kind == "heavy":  # expert 3 in every token + random others
        rest = torch.cat([allx[:3], allx[4:]])
        ids = torch.cat([torch.full((m, 1), 3), pick(rest, TOPK - 1)], 1)
    elif kind == "cold":
        ids = pick(allx[N_HOT:], TOPK)
    elif kind == "hot":
        ids = pick(allx[:N_HOT], TOPK)
    elif kind == "masked":  # random -1 holes, token 0 fully masked
        ids = pick(allx, TOPK)
        ids[torch.rand((m, TOPK), generator=g) < 0.4] = -1
        ids[0] = -1
    elif kind == "shared_only":
        ids = torch.full((m, TOPK), -1)
    elif kind == "one_cold":  # a single cold route in the whole call
        ids = torch.full((m, TOPK), -1)
        ids[m - 1, 0] = N_HOT + 5
    elif kind == "one_hot":
        ids = torch.full((m, TOPK), -1)
        ids[0, TOPK - 1] = 17
    else:
        raise ValueError(kind)
    return ids.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    f = fixture(dev)
    mod, _ = kdev.build(a.v)
    ver = kdev.version(a.v)
    rscale = 2.5 if ver >= 28 else 1.0
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
    e = torch.empty(0, device=dev)
    hm = torch.full((E,), -1, dtype=torch.int32, device=dev)
    cm = torch.full((E,), -1, dtype=torch.int32, device=dev)
    hm[:N_HOT] = torch.arange(N_HOT, dtype=torch.int32, device=dev)
    cm[N_HOT:] = torch.arange(N_COLD, dtype=torch.int32, device=dev)
    kf = {"w13": f["w13"], "w2": f["w2"], "sw13": f["sw13"], "sw2": f["sw2"]}

    def parts(t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    def call(out, x, ids, wt, r, shared=True):
        hot, cold = f["tiers"][r]
        extra = (*f["sslices"][r], SSCALE) if shared else (None, None, 1.0)
        mod.forward(out, x, ids, wt, hm, cm, e, e, e, 0, False, *parts(hot),
                    *parts(cold), ws, True, e, *extra, rscale)

    def err_of(out, ref):
        return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))

    g = torch.Generator().manual_seed(a.seed)
    kinds = ["sparse", "same", "heavy", "cold", "hot", "masked", "shared_only",
             "one_cold", "one_hot"]
    ms = [1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 24, 31, 32]
    scales = {"normal": 0.3, "large": 30.0, "tiny": 3e-4}
    worst, fails, n = 0.0, 0, 0
    cases = []
    for m in ms:
        for kind in kinds:
            cases.append((m, kind, "normal", True))
        for sc in ("large", "tiny"):
            cases.append((m, "sparse", sc, True))
        cases.append((m, "sparse", "zero_rows", True))
        cases.append((m, "sparse", "normal", False))  # no shared expert
    # interleave M and routing so the workspace sees every transition
    order = torch.randperm(len(cases), generator=g).tolist()
    for rep in range(a.reps):
        for ci in order:
            m, kind, sc, shared = cases[ci]
            ids = routing(kind, m, g)
            x = torch.randn((m, HIDDEN), generator=g) * scales.get(sc, 0.3)
            if sc == "zero_rows":
                x[::2] = 0
            x = x.to(torch.bfloat16).to(dev)
            wt = torch.rand((m, TOPK), generator=g).to(dev)
            ids = ids.to(dev)
            for r in range(4):
                ref = kdev.reference(kf, x, ids, wt, shared, SSCALE, r, rscale)
                out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
                call(out, x, ids, wt, r, shared)
                if kind == "shared_only" and not shared:
                    continue
                err = err_of(out, ref)
                bad = not (err <= 5e-3) or bool(torch.isnan(out).any())
                if ref.abs().max() == 0:  # all routes masked, no shared: exact zero
                    bad = bool(out.float().abs().max() != 0)
                    err = float(out.float().abs().max())
                n += 1
                worst = max(worst, err if err == err else 1e9)
                if bad:
                    fails += 1
                    print(json.dumps({"rep": rep, "m": m, "kind": kind, "x": sc,
                                      "shared": shared, "slice": r, "err": err}), flush=True)
    # CUDA graph: several different calls replayed back to back (PDL chain)
    gcases = [(32, "sparse"), (1, "one_cold"), (8, "same"), (17, "masked"),
              (32, "heavy"), (5, "cold")]
    ins, outs, refs = [], [], []
    for m, kind in gcases:
        ids = routing(kind, m, g).to(dev)
        x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16).to(dev)
        wt = torch.rand((m, TOPK), generator=g).to(dev)
        ins.append((x, ids, wt))
        outs.append(torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev))
        refs.append(kdev.reference(kf, x, ids, wt, True, SSCALE, 0, rscale))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (x, ids, wt), o in zip(ins, outs):
            call(o, x, ids, wt, 0)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (x, ids, wt), o in zip(ins, outs):
            call(o, x, ids, wt, 0)
    gfails = 0
    for it in range(50 * a.reps):
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for (m, kind), o, ref in zip(gcases, outs, refs):
            err = err_of(o, ref)
            n += 1
            worst = max(worst, err if err == err else 1e9)
            if not (err <= 5e-3):
                gfails += 1
                print(json.dumps({"graph_iter": it, "m": m, "kind": kind, "err": err}), flush=True)
    fails += gfails
    print(json.dumps({"v": a.v, "calls_checked": n, "worst_rel_err": round(worst, 5),
                      "fails": fails, "graph_fails": gfails, "pass": fails == 0}))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
```
## sliced.py diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/tiered_decode/sliced.py b/vllm/model_executor/layers/fused_moe/tiered_decode/sliced.py
index e47a6e2bc7..e7ddaa11b6 100644
--- a/vllm/model_executor/layers/fused_moe/tiered_decode/sliced.py
+++ b/vllm/model_executor/layers/fused_moe/tiered_decode/sliced.py
@@ -84,6 +84,7 @@ def _library() -> ctypes.CDLL:
         p,
         f,
         f,
+        i,
         p,
     ]
     lib.td_forward.restype = i
@@ -151,6 +152,8 @@ def sliced_decode_moe(
     """
     if (shared_w13 is None) != (shared_w2 is None):
         raise ValueError("Pass both shared expert projections or neither")
+    if hot_map.numel() != cold_map.numel():
+        raise ValueError("hot_map and cold_map must cover the same experts")
     out = torch.empty_like(x)
     ids = topk_ids.to(torch.int32).contiguous()
     wt = topk_weights.float().contiguous()
@@ -171,6 +174,7 @@ def sliced_decode_moe(
         _ptr(shared_w2),
         shared_scale,
         routed_scale,
+        hot_map.numel(),
         torch.cuda.current_stream(x.device).cuda_stream,
     )
     if rc != 0:
```
## sliced_decode.cu (shipped, full)
```cpp
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
    23	// scheduler does not claim past a routed w13 group of several chunks, so a
    24	// CTA never holds w13 work that others could start. The shared expert (bf16
    25	// TP slice) is computed in the same kernel, so a side stream is not needed.
    26	// The output is routed_scale * routed + shared, the MoE's final partial sum.
    27	// Numerics match Marlin's up to fp32 summation order.
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
    59	#ifndef TD_COLD_CTAS
    60	  #define TD_COLD_CTAS 16
    61	#endif
    62	constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
    63	constexpr int MAX_TOK = 8, MAX_TOKENS = 32, MAX_ROUTES = MAX_TOKENS * TOPK,
    64	              MAX_LIST = MAX_ROUTES;
    65	constexpr int R0 = 128, R1 = 128;                // rows per w13 / w2 unit
    66	constexpr int CK0 = 512;                         // w13 K chunk
    67	constexpr int KT0 = CK0 / 16, KT1 = INTER / 16;  // k16 rows per unit
    68	constexpr int G0 = CK0 / 32, G1 = INTER / 32;    // scale groups per unit
    69	constexpr int TILES0 = 2 * INTER / R0, CHUNKS0 = HIDDEN / CK0;
    70	constexpr int UNITS0 = TILES0 * CHUNKS0;  // w13 units per entry
    71	constexpr int TILES1 = HIDDEN / R1;       // w2 units per entry
    72	constexpr int W_BYTES = R0 * CK0 / 2, S_BYTES = G0 * R0 * 2;
    73	static_assert(W_BYTES == R1 * INTER / 2 && S_BYTES == G1 * R1 * 2,
    74	              "w13 and w2 units share the stage layout");
    75	enum Fmt : int { MXFP4 = 0, INT4 = 1 };
    76	constexpr int XROW_BYTES0 = CK0 * 2, XROW_BYTES1 = INTER * 2;
    77	constexpr int XROWS = MAX_TOK;
    78	constexpr int XROW_STRIDE =
    79	    XROW_BYTES0 + 64;  // token rows g, g+1 on disjoint banks
    80	static_assert(XROW_BYTES0 == XROW_BYTES1, "w13 and w2 rows are the same size");
    81	// an x2 row in the workspace: INTER halves, then its fp32 scale (xs2), so one
    82	// bulk copy brings both and the producer has no dependent xs2 load
    83	constexpr int X2_LD = INTER + 8, X2_COPY = XROW_BYTES1 + 16;
    84	static_assert(X2_COPY <= XROW_STRIDE, "x2 row + scale fit a stage row");
    85	// shared expert (bf16, this GPU's 512-wide TP slice of it): units of 256 rows
    86	// x 64 of K, 128 B swizzled, for all T tokens
    87	constexpr int RS = 256, CKS = 64;
    88	constexpr int TILES_S0 = 2 * INTER / RS, CHUNKS_S0 = HIDDEN / CKS;
    89	constexpr int TILES_S1 = HIDDEN / RS, CHUNKS_S1 = INTER / CKS;
    90	constexpr int UNITS_S0 = TILES_S0 * CHUNKS_S0, UNITS_S1 = TILES_S1 * CHUNKS_S1;
    91	static_assert(RS * CKS * 2 == W_BYTES, "a shared unit fills the weight slot");
    92	constexpr int XS_BYTES =
    93	    CKS * 2;  // one token's activation slice per shared unit
    94	// stages start on 1 KB boundaries (128 B swizzle)
    95	constexpr int STAGE_BYTES =
    96	    (W_BYTES + S_BYTES + XROWS * XROW_STRIDE + 1023) / 1024 * 1024;
    97	static_assert(MAX_TOKENS * XS_BYTES <= STAGE_BYTES - W_BYTES,
    98	              "shared rows fit");
    99	constexpr int STAGES = TD_STAGES;
   100	constexpr int CONSUMER_WARPS = 8;
   101	constexpr int THREADS = (CONSUMER_WARPS + 3) * 32;  // + producer, scheduler,
   102	                                                    // finisher
   103	constexpr int SMEM_HEAD = 128;
   104	static_assert(STAGES <= 8, "barrier head holds 8 stages");
   105	constexpr int SMEM_BYTES = SMEM_HEAD + 1024 + STAGES * STAGE_BYTES;
   106	static_assert(SMEM_BYTES <= 227 * 1024, "ring exceeds shared memory");
   107	constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
   108	constexpr int GRID = 132;
   109	constexpr int PREP_THREADS =
   110	    1024;  // route_prep block: 6 hidden elements per thread
   111	static_assert(
   112	    CONSUMER_WARPS == 8,
   113	    "consumer warps are 4 row blocks x 2 K halves (w13), 8 row blocks (w2)");
   114	
   115	struct Expert {
   116	  int local;  // index into the tier's tensors
   117	  int ntok;
   118	  int tok[MAX_TOK];    // token rows
   119	  int route[MAX_TOK];  // token * TOPK + k
   120	  float wt[MAX_TOK];   // router weight * routed_scale
   121	};
   122	
   123	// Per tier and projection: the weight and scale tensor maps. A weight map is
   124	// [E][K/16][N*2] int32 with a {128, 64, 1} box: one 64-row tile's 64 k16 rows,
   125	// 512 B each, landing contiguous in shared memory. A scale map is [E][K/32][N]
   126	// bytes with a {64, 32, 1} box.
   127	struct alignas(64) Tier {
   128	  CUtensorMap w[2];
   129	  CUtensorMap s[2];
   130	};
   131	
   132	// Device workspace, zeroed once at allocation; y13 and y are re-zeroed by
   133	// their last reader so every call finds them clean.
   134	struct Workspace {
   135	  Expert lists[2][MAX_LIST];  // hot, cold
   136	  int counts[2];
   137	  int live[MAX_ROUTES];  // route runs an expert on this GPU
   138	  float xs13[MAX_TOKENS];
   139	  float xs2[MAX_ROUTES];
   140	  alignas(128)
   141	      __half x13[MAX_TOKENS * HIDDEN];  // TMA sources: 16 B aligned at least
   142	  alignas(128) __half x2[MAX_ROUTES * X2_LD];
   143	  alignas(128) float y13[MAX_ROUTES * 2 * INTER];
   144	  alignas(128) float y[MAX_TOKENS * HIDDEN];
   145	  int done13[2]
   146	            [MAX_LIST];    // w13 chunks flushed per entry, zeroed by route_prep
   147	  int ready[2][MAX_LIST];  // epoch once the entry's activation rows are written
   148	  int epoch;               // bumped by route_prep every call
   149	  alignas(128) __nv_bfloat16 x13b[MAX_TOKENS * HIDDEN];  // shared expert input
   150	  alignas(128) __nv_bfloat16 x2s[MAX_TOKENS * INTER];    // its activation
   151	  alignas(128) float y13s[MAX_TOKENS * 2 * INTER];
   152	  int next[2];  // per tier: next group to claim (hot, cold), zeroed by
   153	                // route_prep
   154	  int done_s;   // shared w13 chunks flushed
   155	  int ready_s;  // epoch once x2s is written
   156	  int T;
   157	};
   158	static_assert(offsetof(Workspace, x13) % 16 == 0 &&
   159	                  offsetof(Workspace, x2) % 16 == 0,
   160	              "TMA alignment");
   161	
   162	struct Params {
   163	  Tier tier[2];
   164	  CUtensorMap
   165	      sw[2];  // shared expert w13 [2 * INTER][HIDDEN], w2 [HIDDEN][INTER]
   166	  Workspace* ws;
   167	  float shared_scale;
   168	  int has_shared;
   169	};
   170	
   171	// ---------------------------------------------------------------- PTX helpers
   172	__device__ __forceinline__ uint32_t smem_u32(const void* p) {
   173	  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
   174	}
   175	__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
   176	  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
   177	               "r"(count));
   178	}
   179	__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
   180	  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
   181	                   smem_u32(bar)),
   182	               "r"(bytes)
   183	               : "memory");
   184	}
   185	__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
   186	  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
   187	               : "memory");
   188	}
   189	__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
   190	  uint32_t v;
   191	  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
   192	  return v;
   193	}
   194	__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
   195	  uint4 v;
   196	  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
   197	               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
   198	               : "r"(a));
   199	  return v;
   200	}
   201	__device__ __forceinline__ void mbar_wait_a(uint32_t bar, uint32_t parity) {
   202	  asm volatile(
   203	      "{\n .reg .pred p;\n WAITA_%=:\n "
   204	      "mbarrier.try_wait.parity.shared::cta.b64 "
   205	      "p, [%0], %1;\n"
   206	      " @!p bra WAITA_%=;\n}\n" ::"r"(bar),
   207	      "r"(parity)
   208	      : "memory");
   209	}
   210	__device__ __forceinline__ void mbar_arrive_a(uint32_t bar) {
   211	  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
   212	               : "memory");
   213	}
   214	__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
   215	  asm volatile(
   216	      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
   217	      "p, [%0], %1;\n"
   218	      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
   219	      "r"(parity)
   220	      : "memory");
   221	}
   222	__device__ __forceinline__ void bulk_g2s(void* dst, const void* src,
   223	                                         uint32_t bytes, uint64_t* bar) {
   224	  asm volatile(
   225	      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
   226	      "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
   227	      "l"(src), "r"(bytes), "r"(smem_u32(bar))
   228	      : "memory");
   229	}
   230	__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map,
   231	                                       int c0, int c1, int c2, uint64_t* bar) {
   232	  asm volatile(
   233	      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
   234	      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
   235	      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
   236	      "r"(smem_u32(bar))
   237	      : "memory");
   238	}
   239	__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map,
   240	                                       int c0, int c1, uint64_t* bar) {
   241	  asm volatile(
   242	      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
   243	      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
   244	      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
   245	      : "memory");
   246	}
   247	__device__ __forceinline__ void ldmatrix_x4(uint32_t* a, const void* p) {
   248	  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
   249	               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
   250	               : "r"(smem_u32(p)));
   251	}
   252	__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a,
   253	                                         uint32_t b0, uint32_t b1) {
   254	  asm volatile(
   255	      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
   256	      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
   257	      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
   258	      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
   259	}
   260	__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
   261	                                        uint32_t b0, uint32_t b1,
   262	                                        const float* c) {
   263	  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
   264	      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
   265	      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
   266	      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]),
   267	        "f"(c[1]), "f"(c[2]), "f"(c[3]));
   268	}
   269	__device__ __forceinline__ uint32_t prmt0(uint32_t a, uint32_t sel) {
   270	  uint32_t out;
   271	  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(out) : "r"(a), "r"(sel));
   272	  return out;
   273	}
   274	// One Marlin word -> the four f16x2 A registers. Nibbles, low to high:
   275	// (g,k0) (g,k8) (g+8,k0) (g+8,k8) (g,k1) (g,k9) (g+8,k1) (g+8,k9). Low and high
   276	// nibbles become e5m2 bytes (sign at 7, exponent at 3..2, mantissa at 1), and
   277	// an e5m2 byte is the high byte of the f16 it equals.
   278	__device__ __forceinline__ void decode(uint32_t w, uint32_t* a) {
   279	  const uint32_t lo = ((w << 4) & 0x80808080u) | ((w << 1) & 0x0E0E0E0Eu);
   280	  const uint32_t hi = (w & 0x80808080u) | ((w >> 3) & 0x0E0E0E0Eu);
   281	  a[0] = prmt0(lo, 0x2404);  // row g,   k0 k1
   282	  a[1] = prmt0(lo, 0x3414);  // row g+8, k0 k1
   283	  a[2] = prmt0(hi, 0x2404);  // row g,   k8 k9
   284	  a[3] = prmt0(hi, 0x3414);  // row g+8, k8 k9
   285	}
   286	// The same word as symmetric INT4 (code - 8), in the same fragment order: each
   287	// nibble pair lands under 0x6400 as 1024 + code, and one exact fma gives
   288	// (code - 8) * 2^-14, the scale the MXFP4 decode leaves its values at too.
   289	__device__ __forceinline__ void decode_int4(uint32_t w, uint32_t* a) {
   290	  const __half2 unit = __halves2half2(__ushort_as_half(0x0400),
   291	                                      __ushort_as_half(0x0400));  // 2^-14
   292	  const __half2 bias =
   293	      __halves2half2(__ushort_as_half(0xAC08),
   294	                     __ushort_as_half(0xAC08));  // -1032 * 2^-14
   295	  const int shift[4] = {0, 8, 4, 12};  // rows g, g+8 at k0 k1; then at k8 k9
   296	#pragma unroll
   297	  for (int i = 0; i < 4; ++i) {
   298	    const uint32_t t = ((w >> shift[i]) & 0x000F000Fu) | 0x64006400u;
   299	    const __half2 v =
   300	        __hfma2(*reinterpret_cast<const __half2*>(&t), unit, bias);
   301	    a[i] = *reinterpret_cast<const uint32_t*>(&v);
   302	  }
   303	}
   304	// The same fragment order with one shift and four lop3 per word: nibbles
   305	// (0,4) / (2,6) under 0x6400 give 1024 + code, nibbles (1,5) / (3,7) give
   306	// 1024 + 16 * code; one exact hfma2 each brings both to (code - 8) * 2^-14.
   307	__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t mask,
   308	                                                uint32_t magic) {
   309	  uint32_t d;
   310	  asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
   311	      : "=r"(d)
   312	      : "r"(a), "r"(mask), "r"(magic));
   313	  return d;  // (a & mask) | magic
   314	}
   315	__device__ __forceinline__ void decode_int4_fast(uint32_t w, uint32_t* a) {
   316	  const uint32_t magic = 0x64006400u;
   317	  const uint32_t w8 = w >> 8;
   318	  const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, magic);   // rows g,   k0 k1
   319	  const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, magic);  // rows g+8, k0 k1
   320	  const uint32_t hi0 = lop3_and_or(w, 0x00F000F0u, magic);   // rows g,   k8 k9
   321	  const uint32_t hi1 = lop3_and_or(w8, 0x00F000F0u, magic);  // rows g+8, k8 k9
   322	  const __half2 unit =
   323	      __halves2half2(__ushort_as_half(0x0400), __ushort_as_half(0x0400));
   324	  const __half2 bias =
   325	      __halves2half2(__ushort_as_half(0xAC08), __ushort_as_half(0xAC08));
   326	  // (1024 + 16 c) * 2^-18 - 72 * 2^-14 = (c - 8) * 2^-14
   327	  const __half2 unit16 =
   328	      __halves2half2(__ushort_as_half(0x0040), __ushort_as_half(0x0040));
   329	  const __half2 bias16 =
   330	      __halves2half2(__ushort_as_half(0x9C80), __ushort_as_half(0x9C80));
   331	  __half2 v;
   332	  v = __hfma2(*reinterpret_cast<const __half2*>(&lo0), unit, bias);
   333	  a[0] = *reinterpret_cast<const uint32_t*>(&v);
   334	  v = __hfma2(*reinterpret_cast<const __half2*>(&lo1), unit, bias);
   335	  a[1] = *reinterpret_cast<const uint32_t*>(&v);
   336	  v = __hfma2(*reinterpret_cast<const __half2*>(&hi0), unit16, bias16);
   337	  a[2] = *reinterpret_cast<const uint32_t*>(&v);
   338	  v = __hfma2(*reinterpret_cast<const __half2*>(&hi1), unit16, bias16);
   339	  a[3] = *reinterpret_cast<const uint32_t*>(&v);
   340	}
   341	// Programmatic dependent launch: every kernel of a layer is launched early and
   342	// waits here before touching what its predecessor writes; it releases its own
   343	// successor right after, so each launch and prologue hides behind the previous
   344	// kernel instead of following it.
   345	__device__ __forceinline__ void pdl_wait() {
   346	  asm volatile("griddepcontrol.wait;" ::: "memory");
   347	}
   348	__device__ __forceinline__ void pdl_release() {
   349	  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
   350	}
   351	__device__ __forceinline__ float e8m0(uint32_t byte) {
   352	  return __uint_as_float(byte << 23);
   353	}
   354	
   355	// Element k of an activation row -> its slot in the f16 B-fragment layout:
   356	// [k/32 group][tq][k16 block within group][b0 lo, b0 hi, b1 lo, b1 hi].
   357	__device__ __forceinline__ int frag_slot(int k) {
   358	  const int kb = k / 16, r = k % 16, tq = (r % 8) / 2, reg = r / 8, h = r % 2;
   359	  return (((kb / 2) * 4 + tq) * 2 + kb % 2) * 4 + reg * 2 + h;
   360	}
   361	
   362	// Scale a row so its largest magnitude is 2^13 or below and store it in f16;
   363	// the power of two (times the weights' 2^14) goes to *scale.
   364	__device__ __forceinline__ float row_scale(float m, float* scale) {
   365	  const int t = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
   366	  *scale = exp2f(static_cast<float>(t + PLACEMENT_EXP));
   367	  return exp2f(static_cast<float>(-t));
   368	}
   369	
   370	__device__ __forceinline__ float block_max(float m, float* red) {
   371	  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   372	  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
   373	  __syncthreads();
   374	  if (threadIdx.x < 32) {
   375	    m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
   376	    for (int o = 16; o; o >>= 1)
   377	      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   378	    if (threadIdx.x == 0) red[0] = m;
   379	  }
   380	  __syncthreads();
   381	  m = red[0];
   382	  __syncthreads();
   383	  return m;
   384	}
   385	
   386	// ---------------------------------------------------------------- 1: route +
   387	// prep Blocks [0, T): token rows -> f16 fragments. Block T: which GPU runs
   388	// each active expert (below), then this GPU's per-tier expert lists.
   389	//
   390	// Replica assignment. A hot expert runs on its primary GPU; an active cold
   391	// expert with a replica may run on either holder, and every GPU derives the
   392	// same choice from the same router output. The choice balances predicted
   393	// layer time: COST_US is this path's measured graph-replay time per layer
   394	// (GH200, MiMo-V2 shapes) with h hot and c cold experts on a GPU, made
   395	// non-decreasing. Path reversal moves one cold expert at a time off the
   396	// slowest GPU, along at most three replica hops (each hop is a different GPU
   397	// pair, so the flexible experts are tracked as counts per pair), to the end
   398	// that stays fastest, while that end stays below the source's time.
   399	constexpr int EP = 4, PAIRS = EP * (EP - 1) / 2, MAX_EXPERTS = 512,
   400	              MAX_REVERSALS = 64, COST_HOT = 25, COST_COLD = 7;
   401	__constant__ unsigned short COST_US[COST_HOT][COST_COLD] = {
   402	    {0, 62, 111, 161, 226, 269, 305},    {20, 64, 113, 163, 228, 271, 307},
   403	    {30, 65, 115, 165, 229, 272, 308},   {39, 67, 115, 165, 229, 273, 308},
   404	    {45, 67, 115, 165, 230, 273, 309},   {52, 67, 115, 165, 230, 273, 309},
   405	    {58, 69, 116, 165, 230, 273, 309},   {66, 74, 116, 166, 230, 273, 309},
   406	    {73, 84, 116, 166, 231, 273, 309},   {82, 90, 117, 166, 231, 273, 309},
   407	    {87, 102, 118, 166, 231, 273, 309},  {92, 112, 126, 167, 231, 274, 309},
   408	    {100, 120, 128, 167, 231, 274, 309}, {108, 128, 136, 169, 231, 274, 309},
   409	    {116, 139, 151, 169, 231, 274, 309}, {126, 147, 161, 175, 231, 274, 309},
   410	    {136, 155, 171, 180, 232, 275, 309}, {142, 159, 180, 187, 232, 275, 309},
   411	    {147, 164, 189, 193, 233, 275, 309}, {156, 173, 198, 204, 233, 276, 310},
   412	    {164, 181, 207, 215, 233, 277, 310}, {172, 192, 217, 225, 236, 277, 311},
   413	    {180, 203, 226, 235, 239, 278, 312}, {188, 213, 236, 245, 245, 279, 313},
   414	    {196, 224, 246, 254, 254, 279, 314},
   415	};
   416	
   417	struct Placement {
   418	  const int* primary;      // [E] GPU holding the expert's own copy
   419	  const int* secondary;    // [E] GPU holding a cold replica, or -1
   420	  const int* primary_hot;  // [E] 1 when the own copy is in the hot tier
   421	  int num_experts;         // 0: the slot maps already say what runs here
   422	  int ep_rank;
   423	  bool schedule;  // false: every expert runs on its primary
   424	};
   425	
   426	__device__ __forceinline__ int pair_index(int lo, int hi) {
   427	  return lo * EP - lo * (lo + 1) / 2 + (hi - lo - 1);
   428	}
   429	
   430	__device__ __forceinline__ int cost(const unsigned short (*tab)[COST_COLD],
   431	                                    int h, int c) {
   432	  return tab[min(h, COST_HOT - 1)][min(c, COST_COLD - 1)] +
   433	         8 * max(h - (COST_HOT - 1), 0) + 40 * max(c - (COST_COLD - 1), 0);
   434	}
   435	
   436	// One warp. at_low[k] of the total[k] flexible experts of pair k sit on the
   437	// pair's lower GPU; on return (lane 0) at_low holds the balanced split. Each
   438	// iteration, lane l < 15 tries the l-th path from the slowest GPU in the
   439	// reference order (length, then ascending GPU ids): its target, and the pairs
   440	// it needs an edge on the lower (need_lo) or upper (need_hi) GPU of.
   441	__device__ void reverse_paths(const unsigned short (*tab)[COST_COLD],
   442	                              const int* hot, const int* fixed,
   443	                              const int* total, int* at_low) {
   444	  const int lane = threadIdx.x % 32;
   445	  int split[PAIRS], tot[PAIRS];
   446	#pragma unroll
   447	  for (int k = 0; k < PAIRS; ++k) {
   448	    split[k] = at_low[k];
   449	    tot[k] = total[k];
   450	  }
   451	  int h[EP], f[EP];
   452	#pragma unroll
   453	  for (int r = 0; r < EP; ++r) {
   454	    h[r] = hot[r];
   455	    f[r] = fixed[r];
   456	  }
   457	  for (int step = 0; step < MAX_REVERSALS; ++step) {
   458	    int now[EP], after[EP];
   459	#pragma unroll
   460	    for (int r = 0; r < EP; ++r) {
   461	      int c = f[r];
   462	#pragma unroll
   463	      for (int o = 0; o < EP; ++o)
   464	        if (o != r) {
   465	          const int k = pair_index(min(r, o), max(r, o));
   466	          c += r < o ? split[k] : tot[k] - split[k];
   467	        }
   468	      now[r] = cost(tab, h[r], c);
   469	      after[r] = cost(tab, h[r], c + 1);
   470	    }
   471	    int src = 0;
   472	#pragma unroll
   473	    for (int r = 1; r < EP; ++r)
   474	      if (now[r] > now[src]) src = r;
   475	    // this lane's path; the i-th GPU other than src, ascending, is other(i)
   476	    auto other = [src](int i) { return i < src ? i : i + 1; };
   477	    int hop1 = -1, hop2 = -1, hop3 = -1, len = 0;
   478	    if (lane < 3) {
   479	      len = 1;
   480	      hop1 = other(lane);
   481	    } else if (lane < 15) {
   482	      const int m = lane < 9 ? lane - 3 : lane - 9;
   483	      const int first = m / 2, second = m % 2 < first ? m % 2 : m % 2 + 1;
   484	      len = lane < 9 ? 2 : 3;
   485	      hop1 = other(first);
   486	      hop2 = other(second);
   487	      hop3 = other(3 - first - second);
   488	    }
   489	    const int target = len == 1 ? hop1 : len == 2 ? hop2 : hop3;
   490	    unsigned need_lo = 0, need_hi = 0;
   491	    auto need = [&](int u, int v) {
   492	      const unsigned bit = 1u << pair_index(min(u, v), max(u, v));
   493	      if (u < v)
   494	        need_lo |= bit;
   495	      else
   496	        need_hi |= bit;
   497	    };
   498	    if (len >= 1) need(src, hop1);
   499	    if (len >= 2) need(hop1, hop2);
   500	    if (len >= 3) need(hop2, hop3);
   501	    unsigned have_lo = 0, have_hi = 0;
   502	#pragma unroll
   503	    for (int k = 0; k < PAIRS; ++k) {
   504	      have_lo |= (split[k] > 0 ? 1u : 0u) << k;
   505	      have_hi |= (tot[k] - split[k] > 0 ? 1u : 0u) << k;
   506	    }
   507	    int end_v = 0, src_v = 0;
   508	#pragma unroll
   509	    for (int r = 0; r < EP; ++r) {
   510	      if (r == target) end_v = after[r];
   511	      if (r == src) src_v = now[r];
   512	    }
   513	    const bool usable = len > 0 && (need_lo & ~have_lo) == 0 &&
   514	                        (need_hi & ~have_hi) == 0 && end_v < src_v;
   515	    const unsigned key =
   516	        usable ? static_cast<unsigned>(end_v) << 5 | lane : 0xffffffffu;
   517	    const unsigned best = __reduce_min_sync(0xffffffffu, key);
   518	    if (best == 0xffffffffu) break;
   519	    const int from = best & 31;
   520	    need_lo = __shfl_sync(0xffffffffu, need_lo, from);
   521	    need_hi = __shfl_sync(0xffffffffu, need_hi, from);
   522	#pragma unroll
   523	    for (int k = 0; k < PAIRS; ++k)
   524	      split[k] += ((need_hi >> k) & 1) - ((need_lo >> k) & 1);
   525	  }
   526	  if (lane == 0)
   527	#pragma unroll
   528	    for (int k = 0; k < PAIRS; ++k) at_low[k] = split[k];
   529	}
   530	
   531	template <typename IdT>
   532	__global__ void route_prep_kernel(Workspace* ws, const __nv_bfloat16* x,
   533	                                  const IdT* topk_ids, const bool* padding,
   534	                                  const float* topk_weights, const int* hot_map,
   535	                                  const int* cold_map, Placement pl,
   536	                                  int num_tokens, int hot_size, int cold_size,
   537	                                  int num_experts, float routed_scale) {
   538	  // [hot_size + cold_size] each: tier slot -> its routes here, its first entry
   539	  extern __shared__ int n_of[];
   540	  int* base_of = n_of + hot_size + cold_size;
   541	  __shared__ float red[32];
   542	  pdl_wait();
   543	  pdl_release();
   544	  if (blockIdx.x < num_tokens) {
   545	    // every element loads before any is used: these small kernels are latency
   546	    // bound
   547	    constexpr int PER = HIDDEN / PREP_THREADS;
   548	    const int t = blockIdx.x;
   549	    const __nv_bfloat16* __restrict__ xr = x + static_cast<size_t>(t) * HIDDEN;
   550	    float v[PER], m = 0.f;
   551	#pragma unroll
   552	    for (int i = 0; i < PER; ++i)
   553	      v[i] = __bfloat162float(xr[threadIdx.x + i * PREP_THREADS]);
   554	#pragma unroll
   555	    for (int i = 0; i < PER; ++i) m = fmaxf(m, fabsf(v[i]));
   556	    const float inv = row_scale(block_max(m, red), &ws->xs13[t]);
   557	    __half* __restrict__ out = ws->x13 + static_cast<size_t>(t) * HIDDEN;
   558	#pragma unroll
   559	    for (int i = 0; i < PER; ++i)
   560	      out[frag_slot(threadIdx.x + i * PREP_THREADS)] =
   561	          __float2half_rn(v[i] * inv);
   562	    __nv_bfloat16* __restrict__ outb =
   563	        ws->x13b + static_cast<size_t>(t) * HIDDEN;
   564	#pragma unroll
   565	    for (int i = 0; i < PER; ++i)
   566	      outb[frag_slot(threadIdx.x + i * PREP_THREADS)] =
   567	          xr[threadIdx.x + i * PREP_THREADS];
   568	    return;
   569	  }
   570	  __shared__ int count[2];
   571	  for (int i = threadIdx.x; i < 2 * MAX_LIST; i += blockDim.x)
   572	    (&ws->done13[0][0])[i] = 0;
   573	  // per expert: 1 when routed, then (tier << 16 | slot) where it runs here
   574	  __shared__ int state[MAX_EXPERTS];
   575	  __shared__ int hot_n[EP], fixed_n[EP], total[PAIRS], at_low[PAIRS];
   576	  __shared__ int warp_n[PREP_THREADS / 32][PAIRS];
   577	  __shared__ unsigned short tab[COST_HOT][COST_COLD];
   578	  const int E = pl.num_experts, routes = num_tokens * TOPK, r = threadIdx.x;
   579	  for (int i = threadIdx.x; i < hot_size + cold_size; i += blockDim.x)
   580	    n_of[i] = 0;
   581	  if (threadIdx.x < 2) count[threadIdx.x] = 0;
   582	  if (r < MAX_ROUTES) ws->live[r] = 0;
   583	  // every global read is issued up front: this block is a latency chain
   584	  int e = -1;
   585	  float wt = 0.f;
   586	  if (r < routes) {
   587	    e = static_cast<int>(topk_ids[r]);
   588	    // a padded token's routes are dropped, as if its ids were -1
   589	    if (padding != nullptr && padding[r / TOPK]) e = -1;
   590	    wt = topk_weights[r] * routed_scale;
   591	    if (e >= 0 && E > 0 && e >= E) e = -1;
   592	  }
   593	  // without a placement the slot maps are read with the ids, not after them
   594	  __shared__ int maps[2 * MAX_EXPERTS];
   595	  if (E == 0)
   596	    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
   597	      maps[i] = hot_map[i];
   598	      maps[MAX_EXPERTS + i] = cold_map[i];
   599	    }
   600	  int tier = -1, local = -1, slot = -1;
   601	  if (E > 0) {
   602	    const int x_e = threadIdx.x;
   603	    int p = -1, s = -1, hm = -1, cm = -1;
   604	    bool hot = false;
   605	    if (x_e < E) {
   606	      p = pl.primary[x_e];
   607	      hot = pl.primary_hot[x_e] != 0;
   608	      s = pl.secondary[x_e];
   609	      hm = hot_map[x_e];
   610	      cm = cold_map[x_e];
   611	      state[x_e] = 0;
   612	    }
   613	    if (x_e < EP) hot_n[x_e] = fixed_n[x_e] = 0;
   614	    if (x_e < PAIRS) total[x_e] = at_low[x_e] = 0;
   615	    if (x_e < COST_HOT * COST_COLD)
   616	      tab[x_e / COST_COLD][x_e % COST_COLD] =
   617	          COST_US[x_e / COST_COLD][x_e % COST_COLD];
   618	    __syncthreads();
   619	    if (e >= 0) state[e] = 1;
   620	    __syncthreads();
   621	    int k = -1;
   622	    if (x_e < E && state[x_e]) {
   623	      if (hot)
   624	        atomicAdd(&hot_n[p], 1);
   625	      else if (s < 0)
   626	        atomicAdd(&fixed_n[p], 1);
   627	      else {
   628	        k = pair_index(min(p, s), max(p, s));
   629	        atomicAdd(&total[k], 1);
   630	        if (p < s) atomicAdd(&at_low[k], 1);
   631	      }
   632	    } else {
   633	      p = -1;
   634	    }
   635	    // a flexible expert's position among its pair's, in ascending expert id
   636	    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
   637	    int pos = 0;
   638	#pragma unroll
   639	    for (int j = 0; j < PAIRS; ++j) {
   640	      const unsigned ballot = __ballot_sync(0xffffffffu, k == j);
   641	      if (k == j) pos = __popc(ballot & ((1u << lane) - 1));
   642	      if (lane == 0) warp_n[warp][j] = __popc(ballot);
   643	    }
   644	    __syncthreads();
   645	    if (k >= 0)
   646	      for (int w = 0; w < warp; ++w) pos += warp_n[w][k];
   647	    if (threadIdx.x < 32 && pl.schedule)
   648	      reverse_paths(tab, hot_n, fixed_n, total, at_low);
   649	    __syncthreads();
   650	    if (p >= 0) {
   651	      const int runs_on = k < 0 ? p : pos < at_low[k] ? min(p, s) : max(p, s);
   652	      int code = -1;
   653	      if (runs_on == pl.ep_rank && hot && hm >= 0)
   654	        code = hm;
   655	      else if (runs_on == pl.ep_rank && !hot && cm >= 0)
   656	        code = 1 << 16 | cm;
   657	      state[x_e] = code;
   658	    }
   659	    __syncthreads();
   660	    if (e >= 0 && state[e] >= 0) {
   661	      tier = state[e] >> 16;
   662	      local = state[e] & 0xffff;
   663	    }
   664	  } else {
   665	    __syncthreads();
   666	    if (e >= 0) {
   667	      const int h = maps[e], c = maps[MAX_EXPERTS + e];
   668	      if (h >= 0) {
   669	        tier = 0;
   670	        local = h;
   671	      } else if (c >= 0) {
   672	        tier = 1;
   673	        local = c;
   674	      }
   675	    }
   676	  }
   677	  // each route's position among its expert's; the first route then opens
   678	  // ceil(n / MAX_TOK) consecutive entries, MAX_TOK routes each
   679	  const int idx = tier * hot_size + local;
   680	  int pos = -1;
   681	  if (tier >= 0) pos = atomicAdd(&n_of[idx], 1);
   682	  __syncthreads();
   683	  if (pos == 0) {
   684	    const int n = n_of[idx], entries = (n + MAX_TOK - 1) / MAX_TOK;
   685	    const int base = atomicAdd(&count[tier], entries);
   686	    base_of[idx] = base;
   687	    for (int q = 0; q < entries; ++q) {
   688	      Expert& ex = ws->lists[tier][base + q];
   689	      ex.local = local;
   690	      ex.ntok = min(MAX_TOK, n - q * MAX_TOK);
   691	    }
   692	  }
   693	  __syncthreads();
   694	  if (tier >= 0) {
   695	    slot = base_of[idx] + pos / MAX_TOK;
   696	    Expert& ex = ws->lists[tier][slot];
   697	    const int i = pos % MAX_TOK;
   698	    ex.tok[i] = r / TOPK;
   699	    ex.route[i] = r;
   700	    ex.wt[i] = wt;
   701	    ws->live[r] = 1;
   702	  }
   703	  __syncthreads();
   704	  if (threadIdx.x < 2) ws->counts[threadIdx.x] = count[threadIdx.x];
   705	  if (threadIdx.x == 0) {
   706	    ws->done_s = 0;
   707	    ws->next[0] = ws->next[1] = 0;
   708	    ws->T = num_tokens;
   709	    ws->epoch += 1;
   710	  }
   711	}
   712	
   713	// ---------------------------------------------------------------- 2: the
   714	// layer, one persistent CTA per SM. CTAs [0, cold_ctas) run the cold tier,
   715	// the rest the hot tier. Each CTA runs its share of its tier's w13 units, then
   716	// its share of the w2 units. An entry's w13 chunks are counted as they flush;
   717	// the CTA whose flush completes an entry runs silu * up for its routes and
   718	// releases ready[entry]. A w2 unit's producer issues the weights first and
   719	// waits on ready only before loading the activation rows.
   720	__device__ __forceinline__ int cold_ctas_for(int n_hot, int n_cold) {
   721	  if (n_cold == 0) return 0;
   722	  return n_hot == 0 ? GRID : TD_COLD_CTAS;
   723	}
   724	__device__ __forceinline__ int ld_acquire(const int* p) {
   725	  int v;
   726	  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
   727	               : "=r"(v)
   728	               : "l"(p)
   729	               : "memory");
   730	  return v;
   731	}
   732	__device__ __forceinline__ void st_release(int* p, int v) {
   733	  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v)
   734	               : "memory");
   735	}
   736	__device__ __forceinline__ void fence_proxy_async() {
   737	  asm volatile("fence.proxy.async.global;" ::: "memory");
   738	}
   739	// predicated fire-and-forget add: no branch per element
   740	__device__ __forceinline__ void red_add_if(float* a, float v, bool p) {
   741	  asm volatile(
   742	      "{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q red.global.add.f32 [%0], "
   743	      "%1;\n}" ::"l"(a),
   744	      "f"(v), "r"(static_cast<int>(p))
   745	      : "memory");
   746	}
   747	__device__ __forceinline__ void red_add_v4(float* a, float4 v) {
   748	  asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};" ::"l"(a),
   749	               "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
   750	               : "memory");
   751	}
   752	__device__ __forceinline__ void sts_f32(uint32_t a, float v) {
   753	  asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v));
   754	}
   755	__device__ __forceinline__ float4 lds_f4(uint32_t a) {
   756	  float4 v;
   757	  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
   758	               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
   759	               : "r"(a));
   760	  return v;
   761	}
   762	// One warp's 64 rows x ntok tokens of a routed unit (acc * per-token scale)
   763	// added into base + row(token) * stride: staged through the warp's smem
   764	// scratch [token][68] (conflict-free), then vector reds over contiguous rows.
   765	constexpr int SCR_LD = 68;
   766	__device__ __forceinline__ void flush_rows(float (*acc)[4], const float* fs,
   767	                                           uint32_t scr, float* base,
   768	                                           const int* rows4, int stride,
   769	                                           int ntok, int g, int tq, int lane) {
   770	#pragma unroll
   771	  for (int mb = 0; mb < 4; ++mb)
   772	#pragma unroll
   773	    for (int i = 0; i < 4; ++i) {
   774	      sts_f32(
   775	          scr + ((2 * tq + (i & 1)) * SCR_LD + mb * 16 + g + (i >> 1) * 8) * 4,
   776	          acc[mb][i] * fs[i & 1]);
   777	      acc[mb][i] = 0.f;
   778	    }
   779	  __syncwarp();
   780	#pragma unroll
   781	  for (int j = 0; j < MAX_TOK / 2; ++j) {  // token c = lane / 16 + 2 j
   782	    const int k = lane + 32 * j;
   783	    if (k < ntok * 16) {
   784	      const int c = k >> 4, r4 = k & 15;
   785	      const float4 v = lds_f4(scr + (c * SCR_LD + r4 * 4) * 4);
   786	      red_add_v4(base + static_cast<size_t>(rows4[j]) * stride + r4 * 4, v);
   787	    }
   788	  }
   789	  __syncwarp();
   790	}
   791	__device__ __forceinline__ void fence_acq_rel_gpu() {
   792	  asm volatile("fence.acq_rel.gpu;" ::: "memory");
   793	}
   794	#ifndef TD_FIN_FENCE
   795	  #define TD_FIN_FENCE fence_acq_rel_gpu
   796	#endif
   797	__device__ __forceinline__ void consumer_sync() {
   798	  asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
   799	}
   800	
   801	// silu(gate) * up for one entry's routes, one warp per route: f16 fragments
   802	// of x2 with the row's power-of-two scale; the y13 rows are zeroed for the
   803	// next call.
   804	__device__ __forceinline__ void activate_route(Workspace* ws, int r, int lane) {
   805	  float* __restrict__ yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
   806	  float a[INTER / 32], m = 0.f;
   807	#pragma unroll
   808	  for (int q = 0; q < INTER / 32; ++q) {
   809	    const float g = __ldcg(yr + q * 32 + lane),
   810	                u = __ldcg(yr + INTER + q * 32 + lane);
   811	    a[q] = __fdividef(g, 1.f + __expf(-g)) * u;
   812	    m = fmaxf(m, fabsf(a[q]));
   813	  }
   814	  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   815	  float scale;
   816	  const float inv = row_scale(m, &scale);
   817	  if (lane == 0) ws->xs2[r] = scale;
   818	  __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
   819	  if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
   820	#pragma unroll
   821	  for (int q = 0; q < INTER / 32; ++q) {
   822	    out[frag_slot(q * 32 + lane)] = __float2half_rn(a[q] * inv);
   823	    yr[q * 32 + lane] = 0.f;
   824	    yr[INTER + q * 32 + lane] = 0.f;
   825	  }
   826	}
   827	
   828	// activate_route for the n routes in lanes [0, n) of routel, K at a time
   829	template <int K>
   830	__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
   831	                                                int n, int lane,
   832	                                                uint64_t* tl = nullptr) {
   833	  for (int r0 = 0; r0 < n; r0 += K) {
   834	    float a[K][INTER / 32], m[K];
   835	    float* yr[K];
   836	#pragma unroll
   837	    for (int j = 0; j < K; ++j) {
   838	      const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
   839	      m[j] = 0.f;
   840	      if (r0 + j < n) {
   841	        yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
   842	#pragma unroll
   843	        for (int q = 0; q < INTER / 32; ++q) {
   844	          const float g = __ldcg(yr[j] + q * 32 + lane),
   845	                      u = __ldcg(yr[j] + INTER + q * 32 + lane);
   846	          a[j][q] = __fdividef(g, 1.f + __expf(-g)) * u;
   847	          m[j] = fmaxf(m[j], fabsf(a[j][q]));
   848	        }
   849	      }
   850	    }
   851	#pragma unroll
   852	    for (int j = 0; j < K; ++j) {
   853	      if (r0 + j >= n) break;
   854	      for (int o = 16; o; o >>= 1)
   855	        m[j] = fmaxf(m[j], __shfl_xor_sync(0xffffffffu, m[j], o));
   856	      float scale;
   857	      const float inv = row_scale(m[j], &scale);
   858	      const int r = __shfl_sync(0xffffffffu, routel, r0 + j);
   859	      if (lane == 0) ws->xs2[r] = scale;
   860	      __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
   861	      if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
   862	#pragma unroll
   863	      for (int q = 0; q < INTER / 32; ++q) {
   864	        out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
   865	      }
   866	    }
   867	  }
   868	}
   869	
   870	// zero the y13 rows of the n routes in lanes [0, n) of routel for the next
   871	// call
   872	__device__ __forceinline__ void zero_routes(Workspace* ws, int routel, int n,
   873	                                            int lane) {
   874	  for (int j = 0; j < n; ++j) {
   875	    const int r = __shfl_sync(0xffffffffu, routel, j);
   876	    float4* __restrict__ yr =
   877	        reinterpret_cast<float4*>(ws->y13 + static_cast<size_t>(r) * 2 * INTER);
   878	#pragma unroll
   879	    for (int q = 0; q < 2 * INTER / 128; ++q)
   880	      yr[q * 32 + lane] = make_float4(0.f, 0.f, 0.f, 0.f);
   881	  }
   882	}
   883	
   884	// The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
   885	// y13s is zeroed for the next call.
   886	__device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
   887	                                                int lane) {
   888	  for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
   889	    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
   890	    __nv_bfloat16* __restrict__ out =
   891	        ws->x2s + static_cast<size_t>(tok) * INTER;
   892	#pragma unroll
   893	    for (int q = 0; q < INTER / 32; ++q) {
   894	      const int k = q * 32 + lane;
   895	      const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
   896	      out[frag_slot(k)] =
   897	          __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
   898	      yr[k] = 0.f;
   899	      yr[INTER + k] = 0.f;
   900	    }
   901	  }
   902	}
   903	
   904	// One routed unit's 16 group steps for this warp. PH 0 (w13): 4 row blocks x
   905	// 2 K halves of a 64-row x 1024 box; PH 1 (w2): 8 row blocks of a 128-row x
   906	// 512 box. All smem offsets are immediates off two per-warp bases.
   907	// One routed unit for this warp: all 4 row blocks of a 64-row tile over 4
   908	// group steps of K, so each lane's 16 B of a Marlin k16 row (its 4 row blocks)
   909	// is one conflict-free LDS.128, and each activation fragment feeds 4 MMAs.
   910	// w13 unit (64 rows x 1024): warp w takes K steps [4w, 4w + 4); w2 unit (128
   911	// rows x 512): warp w takes 64-row half w / 4, K steps [4 (w % 4), + 4).
   912	template <int PH>
   913	__device__ __forceinline__ void consume_routed(uint32_t wb, uint32_t sb,
   914	                                               uint32_t xb, float (*acc)[4]) {
   915	  constexpr int KROW = PH ? 1024 : 512;  // bytes per k16 row of the box
   916	  constexpr int SROW = PH ? 256 : 128;   // bytes per scale group row
   917	#pragma unroll
   918	  for (int j = 0; j < 4; ++j) {
   919	    const uint4 w0 = lds_v4(wb + (2 * j) * KROW);
   920	    const uint4 w1 = lds_v4(wb + (2 * j + 1) * KROW);
   921	    const uint4 sw = lds_v4(sb + j * SROW);
   922	    const uint4 xv = lds_v4(xb + j * 64);
   923	    const uint32_t w0s[4] = {w0.x, w0.y, w0.z, w0.w},
   924	                   w1s[4] = {w1.x, w1.y, w1.z, w1.w};
   925	    const uint32_t sws[4] = {sw.x, sw.y, sw.z, sw.w};
   926	    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
   927	#pragma unroll
   928	    for (int mb = 0; mb < 4; mb += 2) {
   929	      float d0[4], d1[4];
   930	      uint32_t a0[4], a1[4];
   931	      decode_int4_fast(w0s[mb], a0);
   932	      decode_int4_fast(w0s[mb + 1], a1);
   933	      mma_f16(d0, a0, xv.x, xv.y, zero);
   934	      mma_f16(d1, a1, xv.x, xv.y, zero);
   935	      decode_int4_fast(w1s[mb], a0);
   936	      decode_int4_fast(w1s[mb + 1], a1);
   937	      mma_f16(d0, a0, xv.z, xv.w, d0);
   938	      mma_f16(d1, a1, xv.z, xv.w, d1);
   939	      const float s00 = __uint_as_float(sws[mb] << 16),
   940	                  s01 = __uint_as_float(sws[mb] & 0xFFFF0000u),
   941	                  s10 = __uint_as_float(sws[mb + 1] << 16),
   942	                  s11 = __uint_as_float(sws[mb + 1] & 0xFFFF0000u);
   943	      acc[mb][0] = fmaf(s00, d0[0], acc[mb][0]);
   944	      acc[mb][1] = fmaf(s00, d0[1], acc[mb][1]);
   945	      acc[mb][2] = fmaf(s01, d0[2], acc[mb][2]);
   946	      acc[mb][3] = fmaf(s01, d0[3], acc[mb][3]);
   947	      acc[mb + 1][0] = fmaf(s10, d1[0], acc[mb + 1][0]);
   948	      acc[mb + 1][1] = fmaf(s10, d1[1], acc[mb + 1][1]);
   949	      acc[mb + 1][2] = fmaf(s11, d1[2], acc[mb + 1][2]);
   950	      acc[mb + 1][3] = fmaf(s11, d1[3], acc[mb + 1][3]);
   951	    }
   952	  }
   953	}
   954	
   955	// Work queues. Each tier's work is a list of groups, claimed in order with
   956	// one atomic by whichever CTA's producer is free: the shared expert's w13
   957	// (tile x 12 chunks), the routed w13 (entry x tile, 6 chunks), the shared w2
   958	// (tile, 8 chunks), the routed w2 (entry x 128-row unit). An entry's w13
   959	// groups go out early and to many CTAs at once, so its w2 units rarely wait;
   960	// the queue ends on 1-unit groups, so CTAs finish within a unit of each
   961	// other. A CTA whose own tier's queue is empty takes the other tier's.
   962	enum Kind : int { K_S0 = 0, K_R0 = 1, K_S1 = 2, K_R1 = 3, K_END = 15 };
   963	constexpr int SCH0 = 12;               // shared w13 chunks per group
   964	constexpr int SG0 = CHUNKS_S0 / SCH0;  // groups per shared w13 tile
   965	static_assert(CHUNKS_S0 % SCH0 == 0, "shared w13 groups tile K");
   966	struct Group {
   967	  int kind, x, c0,
   968	      nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
   969	};
   970	// one claimed group, decoded, with its entry record (scheduler -> producer)
   971	struct SchedEntry {
   972	  int kind, x, c0, nch, q, ei, t0, ntok, local, ready;
   973	  int tok[MAX_TOK], route[MAX_TOK];
   974	  float f[MAX_TOK];  // w13: the token's x13 scale; w2: the route weight
   975	};
   976	// routed w2 units per claim (consecutive tiles of one entry): GR1 above
   977	// GR1_T tokens, else 1. Larger claims amortize the producer's claim, record
   978	// and ready round trips (M=16/32 -5..7%); at M=8 they unbalance the tail.
   979	#ifndef TD_GR1
   980	  #define TD_GR1 2
   981	#endif
   982	#ifndef TD_GR1_T
   983	  #define TD_GR1_T 8
   984	#endif
   985	constexpr int GR1 = TD_GR1;
   986	static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
   987	// A routed w13 tile's 12 K chunks go out as R0S adjacent groups, so several
   988	// CTAs share a late tile and its entry's activation is not stuck behind one
   989	// CTA's 12 serial units (the w2 producers spin on it).
   990	#ifndef TD_R0S
   991	  #define TD_R0S 1  // 2-4 measured slower: more w13 flushes
   992	#endif
   993	constexpr int R0S = TD_R0S, R0CH = CHUNKS0 / R0S;
   994	static_assert(CHUNKS0 % R0S == 0, "w13 groups split the K chunks evenly");
   995	// warps 1..7 run at most STAGES units ahead of warp 0 (the ring), so with
   996	// more units than that between w13 flushes no warp can arrive at the handoff
   997	// barrier for the next flush before warp 0 has synced on this one
   998	static_assert(R0CH > STAGES, "w13 handoff barrier generations cannot overlap");
   999	__device__ __forceinline__ int queue_len(int n, bool sh, int g1) {
  1000	  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
  1001	         n * (TILES0 * R0S + TILES1 / g1);
  1002	}
  1003	__device__ __forceinline__ Group group_at(int gi, int n, bool sh, int g1) {
  1004	  const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0 * R0S,
  1005	            ns1 = sh ? TILES_S1 : 0;
  1006	  if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
  1007	  gi -= ns0;
  1008	  if (gi < nr0) return {K_R0, gi / R0S, (gi % R0S) * R0CH, R0CH};
  1009	  gi -= nr0;
  1010	  if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
  1011	  gi -= ns1;
  1012	  return {K_R1, gi, 0, g1};
  1013	}
  1014	
  1015	__global__ void __launch_bounds__(THREADS, 1)
  1016	    layer_kernel(const __grid_constant__ Params p) {
  1017	  extern __shared__ __align__(128) unsigned char smem[];
  1018	  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  1019	  uint64_t* empty = full + STAGES;
  1020	  // 1 KB aligned (128 B swizzle) by an offset into smem, so every load off it
  1021	  // stays a shared-space LDS rather than a generic LD
  1022	  unsigned char* ring =
  1023	      smem + SMEM_HEAD +
  1024	      ((1024u - ((smem_u32(smem) + SMEM_HEAD) & 1023u)) & 1023u);
  1025	  __shared__ int4 desc[STAGES];
  1026	  // per stage, routed units: each token's destination row (w13: its route's
  1027	  // y13 row; w2: its token's y row) and the scale its flush applies
  1028	  // (w13: xs13[token]; w2: the route weight, times the row's xs2 at the
  1029	  // flush), written by the producer
  1030	  __shared__ int sd_row[STAGES][MAX_TOK];
  1031	  __shared__ float sd_f[STAGES][MAX_TOK];
  1032	  __shared__ int s_last;
  1033	  __shared__ SchedEntry sq[TD_SQ];
  1034	  __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
  1035	  __shared__ int4 hq_info[TD_HQ];  // (q, entry, chunks, end)
  1036	  __shared__ uint64_t hq_full[TD_HQ], hq_empty[TD_HQ];
  1037	  __shared__ __align__(16) float scr_all[CONSUMER_WARPS][MAX_TOK * SCR_LD];
  1038	  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  1039	  Workspace* ws = p.ws;
  1040	  if (threadIdx.x == 0) {
  1041	    for (int s = 0; s < STAGES; ++s) {
  1042	      mbar_init(&full[s], 1);
  1043	      mbar_init(&empty[s], CONSUMER_WARPS);
  1044	    }
  1045	    for (int k = 0; k < TD_SQ; ++k) {
  1046	      mbar_init(&sq_full[k], 1);
  1047	      mbar_init(&sq_empty[k], 1);
  1048	    }
  1049	    for (int k = 0; k < TD_HQ; ++k) {
  1050	      mbar_init(&hq_full[k], CONSUMER_WARPS);
  1051	      mbar_init(&hq_empty[k], 1);
  1052	    }
  1053	    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  1054	  }
  1055	  __syncthreads();
  1056	  pdl_wait();  // route_prep's lists, rows and counters
  1057	  pdl_release();
  1058	  const int n_tier[2] = {ws->counts[0], ws->counts[1]};
  1059	  const int epoch = ws->epoch, T = ws->T;
  1060	  const bool sh = p.has_shared;
  1061	  const int g1 = T > TD_GR1_T ? GR1 : 1;
  1062	  const int len[2] = {queue_len(n_tier[0], sh, g1),
  1063	                      queue_len(n_tier[1], false, g1)};
  1064	  const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
  1065	  const int own = static_cast<int>(blockIdx.x) < cold_ctas ? 1 : 0;
  1066	
  1067	  if (warp == CONSUMER_WARPS + 2) {  // finisher warp
  1068	    int k = 0;
  1069	    uint32_t kph = 0;
  1070	    while (true) {
  1071	      if (lane == 0) mbar_wait(&hq_full[k], kph);
  1072	      __syncwarp();
  1073	      const int4 h = hq_info[k];
  1074	      __syncwarp();
  1075	      if (lane == 0) mbar_arrive(&hq_empty[k]);
  1076	      if (++k == TD_HQ) {
  1077	        k = 0;
  1078	        kph ^= 1;
  1079	      }
  1080	      if (h.w) break;
  1081	      const int q = h.x, ei = h.y, nch = h.z;
  1082	      const Expert& e = ws->lists[q][ei];
  1083	      const int n = e.ntok, routel = e.route[lane & (MAX_TOK - 1)];
  1084	      int done = 0;
  1085	      if (lane == 0) {
  1086	        int old;
  1087	        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
  1088	                     : "=r"(old)
  1089	                     : "l"(&ws->done13[q][ei]), "r"(nch)
  1090	                     : "memory");
  1091	        done = old + nch == UNITS0;
  1092	      }
  1093	      done = __shfl_sync(0xffffffffu, done, 0);
  1094	      if (done) {
  1095	        __syncwarp();  // lane 0's acquire (the count) ordered before every
  1096	                       // lane's y13 reads
  1097	        activate_routes<TD_ACTK>(ws, routel, n, lane);
  1098	        __syncwarp();
  1099	        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
  1100	        zero_routes(ws, routel, n, lane);
  1101	      }
  1102	    }
  1103	    return;
  1104	  }
  1105	  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp
  1106	    const float xs13r = lane < T ? ws->xs13[lane] : 0.f;
  1107	    int q = own, n = 0, k = 0;
  1108	    bool stolen = false;
  1109	    int gi = 0;  // this group's claim; the next one is issued before its loads
  1110	    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
  1111	    gi = __shfl_sync(0xffffffffu, gi, 0);
  1112	    while (true) {
  1113	      if (n >= TD_SQ) {  // slot k is free once its last group is issued
  1114	        if (lane == 0) mbar_wait(&sq_empty[k], ((n / TD_SQ) - 1) & 1);
  1115	        __syncwarp();
  1116	      }
  1117	      SchedEntry& e = sq[k];
  1118	      int gn = 0;
  1119	      // a multi-chunk group is not claimed past: its successor is claimed
  1120	      // once the producer has issued it
  1121	      const bool heavy =
  1122	          gi < (q ? len[1] : len[0]) &&
  1123	          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1).kind !=
  1124	              K_R1;
  1125	      if (lane == 0 && !heavy) gn = atomicAdd(&ws->next[q], 1);
  1126	      if (gi >= (q ? len[1] : len[0])) {
  1127	        bool steal = !stolen;
  1128	        if (steal) {
  1129	          stolen = true;
  1130	          q ^= 1;
  1131	          if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
  1132	          gi = __shfl_sync(0xffffffffu, gi, 0);
  1133	          continue;
  1134	        }
  1135	        if (lane == 0) e.kind = K_END;
  1136	        __syncwarp();
  1137	        if (lane == 0) mbar_arrive(&sq_full[k]);
  1138	        break;
  1139	      }
  1140	      const Group gr =
  1141	          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1);
  1142	      int ei = 0, t0 = 0;
  1143	      if (gr.kind == K_R0) {
  1144	        ei = gr.x / TILES0;
  1145	        t0 = gr.x - ei * TILES0;
  1146	      } else if (gr.kind == K_R1) {
  1147	        if (g1 == 1) {
  1148	          ei = gr.x / TILES1;
  1149	          t0 = gr.x - ei * TILES1;
  1150	        } else {
  1151	          ei = gr.x / (TILES1 / GR1);
  1152	          t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;
  1153	        }
  1154	      }
  1155	      int rdy = 0;
  1156	      if (gr.kind == K_R0 || gr.kind == K_R1) {
  1157	        // one round trip: every field at once (lanes past ntok read a valid
  1158	        // slot and are masked later)
  1159	        const Expert& x = ws->lists[q][ei];
  1160	        const int l7 = lane & (MAX_TOK - 1);
  1161	        const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
  1162	                  routel = x.route[l7];
  1163	        const float wtl = x.wt[l7];
  1164	        // the ready probe in flight with the record loads
  1165	        if (lane == 0 && gr.kind == K_R1)
  1166	          rdy = ld_acquire(&ws->ready[q][ei]) == epoch;
  1167	        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
  1168	        if (lane < MAX_TOK) {
  1169	          e.tok[lane] = tokl;
  1170	          e.route[lane] = routel;
  1171	          e.f[lane] = gr.kind == K_R0 ? xs : wtl;
  1172	        }
  1173	        if (lane == 0) {
  1174	          e.ntok = ntok;
  1175	          e.local = local;
  1176	        }
  1177	      }
  1178	      if (lane == 0) {
  1179	        e.kind = gr.kind;
  1180	        e.x = gr.x;
  1181	        e.c0 = gr.c0;
  1182	        e.nch = gr.nch;
  1183	        e.q = q;
  1184	        e.ei = ei;
  1185	        e.t0 = t0;
  1186	        // one probe: if the entry is ready, this acquire is handed to the
  1187	        // producer by the FIFO barrier and its spin is skipped
  1188	        e.ready = rdy;
  1189	      }
  1190	      __syncwarp();
  1191	      if (lane == 0) mbar_arrive(&sq_full[k]);
  1192	      if (heavy && lane == 0) {
  1193	        mbar_wait(&sq_empty[k], (n / TD_SQ) & 1);
  1194	        gn = atomicAdd(&ws->next[q], 1);
  1195	      }
  1196	      if (++k == TD_SQ) k = 0;
  1197	      ++n;
  1198	      gi = __shfl_sync(0xffffffffu, gn, 0);
  1199	    }
  1200	    return;
  1201	  }
  1202	  if (warp == CONSUMER_WARPS) {  // producer warp
  1203	    if (lane == 0) {
  1204	      for (int q = 0; q < 2; ++q)
  1205	        for (int k = 0; k < 2; ++k) {
  1206	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1207	                           reinterpret_cast<uint64_t>(&p.tier[q].w[k]))
  1208	                       : "memory");
  1209	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1210	                           reinterpret_cast<uint64_t>(&p.tier[q].s[k]))
  1211	                       : "memory");
  1212	        }
  1213	      if (sh)
  1214	        for (int k = 0; k < 2; ++k)
  1215	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1216	                           reinterpret_cast<uint64_t>(&p.sw[k]))
  1217	                       : "memory");
  1218	    }
  1219	    int last_ready = -1, it = 0, k = 0;
  1220	    uint32_t kph = 0;
  1221	    const int gi = 0;
  1222	    (void)gi;
  1223	    while (true) {
  1224	      if (lane == 0) mbar_wait(&sq_full[k], kph);
  1225	      __syncwarp();
  1226	      const SchedEntry& e = sq[k];
  1227	      if (e.kind == K_END) break;
  1228	      const Group gr = {e.kind, e.x, e.c0, e.nch};
  1229	      const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = e.ready;
  1230	      int ntok = 0, local = 0, tokl = 0, routel = 0;
  1231	      float fl = 0.f;
  1232	      if (gr.kind == K_R0 || gr.kind == K_R1) {
  1233	        const int l7 = lane & (MAX_TOK - 1);
  1234	        ntok = e.ntok;
  1235	        local = e.local;
  1236	        tokl = e.tok[l7];
  1237	        routel = e.route[l7];
  1238	        fl = e.f[l7];
  1239	      }
  1240	      const Tier& tr = p.tier[q];
  1241	      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
  1242	        const int s = it % STAGES, c = gr.c0 + ci;
  1243	        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
  1244	        if (lane == 0 && it >= STAGES)
  1245	          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
  1246	        __syncwarp();
  1247	        const int hdr = gr.kind | q << 4 | (ci == 0) << 8 |
  1248	                        (ci == gr.nch - 1) << 9 | gr.nch << 16;
  1249	        if (gr.kind == K_R0) {
  1250	          if (lane < ntok) {
  1251	            sd_row[s][lane] = routel;
  1252	            sd_f[s][lane] = fl;
  1253	          }
  1254	          __syncwarp();
  1255	          if (lane == 0) {
  1256	            desc[s] = make_int4(hdr, ei, t0, ntok);
  1257	            mbar_expect_tx(&full[s],
  1258	                           W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
  1259	            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
  1260	            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
  1261	          }
  1262	          __syncwarp();
  1263	          if (lane < ntok && lane < XROWS)
  1264	            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
  1265	                     ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
  1266	                     XROW_BYTES0, &full[s]);
  1267	        } else if (gr.kind == K_R1) {
  1268	          const int t = t0 + ci;
  1269	          if (lane < ntok) {
  1270	            sd_row[s][lane] = tokl;
  1271	            sd_f[s][lane] = fl;
  1272	          }
  1273	          __syncwarp();
  1274	          if (lane == 0) {
  1275	            desc[s] = make_int4(hdr, ei, t, ntok);
  1276	            mbar_expect_tx(&full[s],
  1277	                           W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
  1278	            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
  1279	            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
  1280	          }
  1281	          // weights are in flight; only the activation rows wait for the
  1282	          // entry's ready (every copying lane acquires for itself)
  1283	          if ((ei | q << 16) != last_ready) {
  1284	            if (!pre_ready) {
  1285	              while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
  1286	            }
  1287	            fence_proxy_async();
  1288	            last_ready = ei | q << 16;
  1289	          }
  1290	          __syncwarp();
  1291	          if (lane < ntok && lane < XROWS)
  1292	            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
  1293	                     ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
  1294	                     &full[s]);
  1295	        } else {  // shared expert
  1296	          if (lane == 0) {
  1297	            desc[s] = make_int4(hdr, gr.x, c, 0);
  1298	            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
  1299	            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
  1300	                   &full[s]);
  1301	          }
  1302	          if (gr.kind == K_S1 &&
  1303	              last_ready != -2) {  // every copying lane acquires
  1304	            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
  1305	            fence_proxy_async();
  1306	          }
  1307	          if (gr.kind == K_S1) last_ready = -2;
  1308	          __syncwarp();
  1309	          if (lane < T)
  1310	            bulk_g2s(dst + W_BYTES + lane * XS_BYTES,
  1311	                     (gr.kind == K_S0
  1312	                          ? ws->x13b + static_cast<size_t>(lane) * HIDDEN
  1313	                          : ws->x2s + static_cast<size_t>(lane) * INTER) +
  1314	                         c * CKS,
  1315	                     XS_BYTES, &full[s]);
  1316	        }
  1317	      }
  1318	      __syncwarp();
  1319	      if (lane == 0) mbar_arrive(&sq_empty[k]);
  1320	      if (++k == TD_SQ) {
  1321	        k = 0;
  1322	        kph ^= 1;
  1323	      }
  1324	    }
  1325	    if (lane == 0) {  // end of work
  1326	      const int s = it % STAGES;
  1327	      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
  1328	      desc[s] = make_int4(K_END, 0, 0, 0);
  1329	      mbar_arrive(&full[s]);
  1330	    }
  1331	    return;
  1332	  }
  1333	
  1334	  // consumer warps
  1335	  const int g = lane / 4, tq = lane % 4;
  1336	  const int nt_n = (T + 7) / 8;
  1337	  float acc[4][4] = {};
  1338	  float accs[2][4][4] = {};
  1339	  int hk = 0, hn = 0;  // handoff slot and count (same in every consumer)
  1340	  const auto hand_off = [&](int4 h) {
  1341	    // every warp waits for the slot's previous use to be taken before it
  1342	    // arrives (no arrival may enter a phase whose predecessor nobody waited
  1343	    // on); __syncwarp gathers the warp's reds and warp 0's info for lane 0's
  1344	    // release
  1345	    if (lane == 0) {
  1346	      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
  1347	      if (warp == 0) hq_info[hk] = h;
  1348	    }
  1349	    __syncwarp();
  1350	    if (lane == 0) mbar_arrive(&hq_full[hk]);
  1351	    if (++hk == TD_HQ) hk = 0;
  1352	    ++hn;
  1353	  };
  1354	  // per-warp smem offsets within a stage, computed once
  1355	  const uint32_t ring_u = smem_u32(ring), full_u = smem_u32(full),
  1356	                 empty_u = smem_u32(empty);
  1357	  const uint32_t desc_u = smem_u32(desc), sdf_u = smem_u32(&sd_f[0][2 * tq]);
  1358	  const uint32_t scr_u = smem_u32(scr_all[warp]),
  1359	                 sdr0_u = smem_u32(&sd_row[0][0]);
  1360	  static_assert(CONSUMER_WARPS == 8,
  1361	                "8 K slices (w13), 2 halves x 4 K slices (w2)");
  1362	  const int h1 = warp / 4, k1 = warp % 4;
  1363	  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  1364	  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  1365	  const uint32_t xo1 =
  1366	      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (k1 * 16 + tq) * 16;
  1367	  int s = 0;
  1368	  uint32_t ph = 0;
  1369	  for (int it = 0;; ++it) {
  1370	    mbar_wait_a(full_u + 8 * s, ph);
  1371	    const uint4 du = lds_v4(desc_u + 16 * s);
  1372	    const int4 d = make_int4(du.x, du.y, du.z, du.w);
  1373	    const int kind = d.x & 15;
  1374	    if (kind == K_END) {
  1375	      hand_off(make_int4(0, 0, 0, 1));
  1376	      break;
  1377	    }
  1378	    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
  1379	    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
  1380	    const uint32_t st_u = ring_u + s * STAGE_BYTES;
  1381	    const uint32_t empty_s = empty_u + 8 * s;
  1382	    // this lane's two tokens of a routed unit: destination rows and scales
  1383	    const int ntok = d.w;
  1384	    // flush metadata of stage s (plain shared loads the compiler may schedule
  1385	    // into the math), read before the stage is released
  1386	    const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
  1387	      fs_[0] = sd_f[s_][2 * tq];
  1388	      fs_[1] = sd_f[s_][2 * tq + 1];
  1389	      if (r1) {
  1390	        const unsigned char* xr =
  1391	            ring + static_cast<size_t>(s_) * STAGE_BYTES + W_BYTES + S_BYTES +
  1392	            ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
  1393	        fs_[0] *= *reinterpret_cast<const float*>(xr);
  1394	        fs_[1] *=
  1395	            *reinterpret_cast<const float*>(xr + (XROWS > 1 ? XROW_STRIDE : 0));
  1396	      }
  1397	#pragma unroll
  1398	      for (int j = 0; j < MAX_TOK / 2; ++j)
  1399	        rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
  1400	    };
  1401	    float fs[2];
  1402	    int rows4[MAX_TOK / 2];
  1403	
  1404	    if (kind == K_R0) {
  1405	      // the group's nch chunks in a row: its later stages carry nothing new
  1406	      // for the consumer (same entry, tile, tokens), so no descriptor reads
  1407	      for (int c = 0;; ++c) {
  1408	        const uint32_t su = ring_u + s * STAGE_BYTES;
  1409	        consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
  1410	        if (c == nch - 1) flush_meta(s, fs, rows4, false);
  1411	        __syncwarp();
  1412	        if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
  1413	        if (c == nch - 1) break;
  1414	        if (++s == STAGES) {
  1415	          s = 0;
  1416	          ph ^= 1;
  1417	        }
  1418	        mbar_wait_a(full_u + 8 * s, ph);
  1419	      }
  1420	      {
  1421	        const int ei = d.y, t = d.z;
  1422	        flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
  1423	                   ntok, g, tq, lane);
  1424	        hand_off(make_int4(q, ei, nch, 0));
  1425	      }
  1426	    } else if (kind == K_R1) {
  1427	      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
  1428	      flush_meta(s, fs, rows4, true);
  1429	      __syncwarp();
  1430	      if (lane == 0) mbar_arrive_a(empty_s);
  1431	      const int t = d.z;
  1432	      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
  1433	                 g, tq, lane);
  1434	    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
  1435	      const unsigned char* xa = st + W_BYTES;
  1436	#pragma unroll
  1437	      for (int ks = 0; ks < CKS / 16; ++ks) {
  1438	        uint32_t af[2][4];
  1439	#pragma unroll
  1440	        for (int rb = 0; rb < 2; ++rb) {
  1441	          const int row =
  1442	              warp * 32 + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
  1443	          const int ck = ks * 2 + (lane >> 4);
  1444	          ldmatrix_x4(af[rb], st + row * 128 + ((ck ^ (row & 7)) << 4));
  1445	        }
  1446	#pragma unroll
  1447	        for (int nt = 0; nt < 4; ++nt) {
  1448	          if (nt < nt_n) {
  1449	            const uint2 bv = *reinterpret_cast<const uint2*>(
  1450	                xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 +
  1451	                (ks & 1) * 8);
  1452	            mma_bf16(accs[0][nt], af[0], bv.x, bv.y);
  1453	            mma_bf16(accs[1][nt], af[1], bv.x, bv.y);
  1454	          }
  1455	        }
  1456	      }
  1457	      __syncwarp();
  1458	      if (lane == 0) mbar_arrive_a(empty_s);
  1459	      if (last) {
  1460	        const int t = d.y;
  1461	#pragma unroll
  1462	        for (int rb = 0; rb < 2; ++rb)
  1463	#pragma unroll
  1464	          for (int nt = 0; nt < 4; ++nt)
  1465	#pragma unroll
  1466	            for (int i = 0; i < 4; ++i) {
  1467	              const int tok = nt * 8 + 2 * tq + (i & 1);
  1468	              const int row = t * RS + warp * 32 + rb * 16 + g + (i >> 1) * 8;
  1469	              if (nt < nt_n && tok < T) {
  1470	                if (kind == K_S0)
  1471	                  atomicAdd(
  1472	                      &ws->y13s[static_cast<size_t>(tok) * 2 * INTER + row],
  1473	                      accs[rb][nt][i]);
  1474	                else
  1475	                  atomicAdd(&ws->y[static_cast<size_t>(tok) * HIDDEN + row],
  1476	                            p.shared_scale * accs[rb][nt][i]);
  1477	              }
  1478	              accs[rb][nt][i] = 0.f;
  1479	            }
  1480	        if (kind == K_S0) {
  1481	          __threadfence();  // every thread's own y13s atomics, before the count
  1482	          consumer_sync();
  1483	          if (threadIdx.x == 0) {
  1484	            __threadfence();
  1485	            s_last = atomicAdd(&ws->done_s, nch) + nch == UNITS_S0;
  1486	          }
  1487	          consumer_sync();
  1488	          if (s_last) {
  1489	            __threadfence();
  1490	            activate_shared(ws, T, warp, lane);
  1491	            fence_proxy_async();
  1492	            __threadfence();
  1493	            consumer_sync();
  1494	            if (threadIdx.x == 0) st_release(&ws->ready_s, epoch);
  1495	          }
  1496	        }
  1497	      }
  1498	    }
  1499	    if (++s == STAGES) {
  1500	      s = 0;
  1501	      ph ^= 1;
  1502	    }
  1503	  }
  1504	}
  1505	
  1506	// ---------------------------------------------------------------- 5: finalize
  1507	// One float4 per thread: grid [T][HIDDEN / 1024] x 256.
  1508	__global__ void finalize_kernel(Workspace* ws,
  1509	                                __nv_bfloat16* __restrict__ out) {
  1510	  pdl_wait();
  1511	  const size_t i = (static_cast<size_t>(blockIdx.y) * HIDDEN +
  1512	                    blockIdx.x * 1024 + threadIdx.x * 4);
  1513	  float4* y = reinterpret_cast<float4*>(ws->y + i);
  1514	  const float4 v = *y;
  1515	  *y = make_float4(0.f, 0.f, 0.f, 0.f);
  1516	  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + i);
  1517	  o[0] = __floats2bfloat162_rn(v.x, v.y);
  1518	  o[1] = __floats2bfloat162_rn(v.z, v.w);
  1519	}
  1520	
  1521	// ---------------------------------------------------------------- host entry
  1522	// Plain C ABI (no torch headers: the build is the kernel alone), called from
  1523	// Python through ctypes with raw device pointers and the current stream.
  1524	#define TD_REQUIRE(c, msg)                              \
  1525	  do {                                                  \
  1526	    if (!(c)) {                                         \
  1527	      std::fprintf(stderr, "tiered_decode: %s\n", msg); \
  1528	      return -1;                                        \
  1529	    }                                                   \
  1530	  } while (0)
  1531	
  1532	// [n2][n1][n0] contiguous elements of elem bytes; box {b0, b1, 1}
  1533	static int make_map3(CUtensorMap* map, const void* p, uint64_t n0, uint64_t n1,
  1534	                     uint64_t n2, int elem, CUtensorMapDataType type,
  1535	                     uint32_t b0, uint32_t b1) {
  1536	  const cuuint64_t dims[3] = {n0, n1, n2};
  1537	  const cuuint64_t strides[2] = {n0 * elem, n0 * n1 * elem};
  1538	  const cuuint32_t box[3] = {b0, b1, 1}, unit[3] = {1, 1, 1};
  1539	  return cuTensorMapEncodeTiled(
  1540	             map, type, 3, const_cast<void*>(p), dims, strides, box, unit,
  1541	             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
  1542	             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
  1543	             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
  1544	             ? 0
  1545	             : -1;
  1546	}
  1547	// [rows][cols] bf16 with a {b0, b1} box, 128 B swizzled
  1548	static int make_map_2d_sw128(CUtensorMap* map, const void* p, uint64_t cols,
  1549	                             uint64_t rows, uint32_t b0, uint32_t b1) {
  1550	  const cuuint64_t dims[2] = {cols, rows};
  1551	  const cuuint64_t strides[1] = {cols * 2};
  1552	  const cuuint32_t box[2] = {b0, b1}, unit[2] = {1, 1};
  1553	  return cuTensorMapEncodeTiled(
  1554	             map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p),
  1555	             dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE,
  1556	             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
  1557	             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
  1558	             ? 0
  1559	             : -1;
  1560	}
  1561	static int fill_tier(Tier& tr, const void* w13, const void* s13, const void* w2,
  1562	                     const void* s2, int e) {
  1563	  std::memset(&tr, 0, sizeof(tr));
  1564	  if (e == 0) return 0;
  1565	  int r = make_map3(&tr.w[0], w13, 2 * INTER * 2, HIDDEN / 16, e, 4,
  1566	                    CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R0, KT0);
  1567	  r |= make_map3(&tr.w[1], w2, HIDDEN * 2, INTER / 16, e, 4,
  1568	                 CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R1, KT1);
  1569	  r |= make_map3(&tr.s[0], s13, 2 * INTER, HIDDEN / 32, e, 2,
  1570	                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R0, G0);
  1571	  r |= make_map3(&tr.s[1], s2, HIDDEN, INTER / 32, e, 2,
  1572	                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R1, G1);
  1573	  return r;
  1574	}
  1575	
  1576	}  // namespace tiered_decode
  1577	
  1578	using namespace tiered_decode;
  1579	
  1580	extern "C" long long td_workspace_bytes() { return sizeof(Workspace); }
  1581	
  1582	extern "C" int td_forward(void* out, const void* x, const int* ids,
  1583	                          const float* wt, const int* hot_map,
  1584	                          const int* cold_map, int T, const void* hw13,
  1585	                          const void* hs13, const void* hw2, const void* hs2,
  1586	                          int hot_size, const void* cw13, const void* cs13,
  1587	                          const void* cw2, const void* cs2, int cold_size,
  1588	                          void* workspace, int pdl_launch, const bool* padding,
  1589	                          const void* sw13, const void* sw2, float shared_scale,
  1590	                          float routed_scale, int num_experts,
  1591	                          void* stream_ptr) {
  1592	  TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..32 tokens");
  1593	  TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
  1594	  TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS, "1..512 experts");
  1595	  Params p{};
  1596	  TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
  1597	             "hot tier maps");
  1598	  TD_REQUIRE(fill_tier(p.tier[1], cw13, cs13, cw2, cs2, cold_size) == 0,
  1599	             "cold tier maps");
  1600	  p.ws = reinterpret_cast<Workspace*>(workspace);
  1601	  p.has_shared = sw13 != nullptr;
  1602	  p.shared_scale = shared_scale;
  1603	  if (p.has_shared) {
  1604	    TD_REQUIRE(
  1605	        make_map_2d_sw128(&p.sw[0], sw13, HIDDEN, 2 * INTER, CKS, RS) == 0,
  1606	        "sw13 map");
  1607	    TD_REQUIRE(make_map_2d_sw128(&p.sw[1], sw2, INTER, HIDDEN, CKS, RS) == 0,
  1608	               "sw2 map");
  1609	  }
  1610	  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  1611	  static bool attrs = false;
  1612	  if (!attrs) {
  1613	    cudaFuncSetAttribute(
  1614	        layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  1615	    attrs = true;
  1616	  }
  1617	  cudaLaunchAttribute pdl[1];
  1618	  pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  1619	  pdl[0].val.programmaticStreamSerializationAllowed = 1;
  1620	  auto config = [&](dim3 grid, dim3 block, size_t smem) {
  1621	    cudaLaunchConfig_t c = {};
  1622	    c.gridDim = grid;
  1623	    c.blockDim = block;
  1624	    c.dynamicSmemBytes = smem;
  1625	    c.stream = stream;
  1626	    c.attrs = pdl;
  1627	    c.numAttrs = pdl_launch ? 1 : 0;
  1628	    return c;
  1629	  };
  1630	  Placement pl{};
  1631	  const size_t route_smem =
  1632	      2 * static_cast<size_t>(hot_size + cold_size) * sizeof(int);
  1633	  cudaLaunchConfig_t c = config(dim3(T + 1), dim3(PREP_THREADS), route_smem);
  1634	  if (cudaLaunchKernelEx(&c, route_prep_kernel<int>, p.ws,
  1635	                         reinterpret_cast<const __nv_bfloat16*>(x), ids,
  1636	                         padding, wt, hot_map, cold_map, pl, T, hot_size,
  1637	                         cold_size, num_experts, routed_scale) != cudaSuccess)
  1638	    return -2;
  1639	  c = config(dim3(GRID), dim3(THREADS), SMEM_BYTES);
  1640	  if (cudaLaunchKernelEx(&c, layer_kernel, p) != cudaSuccess) return -3;
  1641	  c = config(dim3(HIDDEN / 1024, T), dim3(256), 0);
  1642	  if (cudaLaunchKernelEx(&c, finalize_kernel, p.ws,
  1643	                         reinterpret_cast<__nv_bfloat16*>(out)) != cudaSuccess)
  1644	    return -4;
  1645	  return 0;
  1646	}
```
