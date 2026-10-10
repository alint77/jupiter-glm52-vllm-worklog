# Astra review 13b: sign-off on the review-13 fixes (sliced MoE decode kernel)

Do NOT run any commands. Your review 13 was "conditional ship". Below:
- the kernel diff that applies your conditions (td_v54 → td_v55, the shipped file);
- the extended adversarial check (full diff);
- results.

Please confirm ship, or name what still blocks it.

## Kernel fixes (diff below)
1. **Dead `TD_FIN_FENCE` / `fence_acq_rel_gpu` removed.** The review-12 decision superseded the
   separate count fence with lane 0's `atom.acq_rel.gpu` (ATOM_AR). So in the shipped default path
   the finisher's ordering is:
   handoff barrier → lane-0 `atom.acq_rel.gpu` count → `__syncwarp` → loads → stores →
   `__syncwarp` → lane-0 `st.release.gpu`.
   No separate fence remains. The knob was left over from the v51 variants.
2. **Map bound.** `if (e >= (E > 0 ? E : num_experts)) e = -1;` at the id read. This covers the
   placement-less path; the placement path is unchanged.
3. **Epoch wrap.** route_prep thread 0 now computes the next epoch in unsigned arithmetic. When it
   wraps to 0, it clears `ready[2][MAX_LIST]` and `ready_s` and uses 1. The readers are in
   layer_kernel after its PDL wait on route_prep, so these stores are visible as the ready-flag
   stores were before.
4. **Comments.** The scheduler comment and header now say the claim waits until the producer has
   issued all chunks of a w13 or shared w2 group.

SASS: layer_kernel and finalize_kernel are byte-identical to td_v54, so the bench and replay
numbers carry over; only route_prep changed (61 regs).

## Adversarial check, round 2 (diff below)
- **Map configs**, chosen at random per case:
  - identity;
  - permuted hot and cold slots, with the tier tensors reordered to match;
  - absent: experts 3, 17, 45 in neither map;
  - empty hot tier: null pointers, size 0, hot map all -1;
  - empty cold tier.
- **The real padding argument** in 25% of cases: random tokens are padded, and the reference drops
  their routes.
- **New routing kinds:**
  - same_cold: every token on the same 8 cold experts;
  - isolate: one-hot router weights with the shared expert off;
  - all routes masked with the shared expert off (the old skip is removed);
  - one_cold / one_hot / masked / sparse with the shared expert off.
- **Per-row check.** For each row: max |out - ref| / max |ref row| ≤ 5e-3. Rows whose reference is
  zero must be exactly 0. Any NaN fails.
- **Graph check.**
  - Chain: 8 calls (M 32/1/8/17/4/32/5/9) with the shared expert both on and off, plus an empty
    call (all masked, no shared expert).
  - Inputs: new random ids, x and weights are copied into the captured buffers before every replay,
    with matching references.
  - 60 replays.
- **Epoch wrap.**
  - Locating the words: the epoch word is the only one an empty call bumps. The ready flags are the
    words a repeated busy call bumps that also equal its epoch.
  - Then: epoch := -1 and flags := 0, i.e. stale flags that alias what a plain increment wraps to.
  - Then: three new M=32 calls, checked, with 10 trials. v55 must end at epoch 3.
- **Out-of-range ids.** 30% of the routes get id E+3 or 700; the reference drops them.
- **Duplicate ids.** All 256 routes on one cold expert (outside top-k's contract); reported only.

## Results, seeds 21-24, reps 3
| kernel | calls/seed | fails | graph fails | wrap fails | oob fails | worst row err | dup_cold err |
|---|--:|--:|--:|--:|--:|--:|--:|
| td_v55 (ship) | 3477 | 0 | 0 | 0 | 0 | 0.00412 | ≤0.0036 |
| td_v54 (no wrap fix) | 3474 | 10-15 | 0 | 10-15 | (n/a) | 0.00401-0.00412 | ≤0.0036 |

In td_v54, every failure is a wrap call: rel err 0.7-1.6 at epoch 0, plus some calls after the wrap.
So the wrap test detects the aliasing it targets.

Cases per config, per seed: ident ~245, perm ~118, absent ~127, nohot ~123, nocold ~127.

Earlier results on the same source:
- Round-1 adversarial check on td_v54: 4 seeds × 4880 calls, 0 fails.
- vLLM `tests/kernels/moe/test_tiered_decode_sliced.py`: 6 passed.

## Performance claims as they will be written
They follow your point 4:
- The replay is modeled: per-cell fits plus held-out agentic routings.
- It predicts the shipped kernel's MoE time at about 20% (M=8) and 16% (M=32) below prod EP's
  slowest GPU, and about 7% below EP's mean GPU.
- The NOAHEAD increment over v53 (0.7% / 1.9%) is not established by paired measurement.

## Kernel diff (td_v54 → shipped)
```diff
--- kernels/td_v54.cu	2026-10-10 14:13:26.057108000 +0200
+++ kernels/td_v55.cu	2026-10-10 14:24:07.664303000 +0200
@@ -20,11 +20,11 @@
 // accumulation. Routed w13 partials go to y13 through fp32 reds; a finisher
 // warp counts each finished w13 group, and the CTA that completes an entry
 // applies silu * up and publishes the entry, whose w2 units then run. The
-// scheduler does not claim past a routed w13 group of several chunks, so a
-// CTA never holds w13 work that others could start. The shared expert (bf16
-// TP slice) is computed in the same kernel, so a side stream is not needed.
-// The output is routed_scale * routed + shared, the MoE's final partial sum.
-// Numerics match Marlin's up to fp32 summation order.
+// scheduler claims the next group only after the producer has issued a w13
+// or shared w2 group, so a CTA never holds w13 work that others could start.
+// The shared expert (bf16 TP slice) is computed in the same kernel, so a side
+// stream is not needed. The output is routed_scale * routed + shared, the MoE's
+// final partial sum. Numerics match Marlin's up to fp32 summation order.
 //
 // C ABI (no torch headers): td_workspace_bytes(), td_forward(...).
 
@@ -260,8 +260,7 @@
 __device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
                                         uint32_t b0, uint32_t b1,
                                         const float* c) {
-  asm(
-      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
+  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
       : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]),
@@ -530,12 +529,12 @@
 }
 
 template <typename IdT>
-__global__ void 
-route_prep_kernel(Workspace* ws, const __nv_bfloat16* x, const IdT* topk_ids,
-                  const bool* padding, const float* topk_weights,
-                  const int* hot_map, const int* cold_map, Placement pl,
-                  int num_tokens, int hot_size, int cold_size,
-                  int num_experts, float routed_scale) {
+__global__ void route_prep_kernel(Workspace* ws, const __nv_bfloat16* x,
+                                  const IdT* topk_ids, const bool* padding,
+                                  const float* topk_weights, const int* hot_map,
+                                  const int* cold_map, Placement pl,
+                                  int num_tokens, int hot_size, int cold_size,
+                                  int num_experts, float routed_scale) {
   // [hot_size + cold_size] each: tier slot -> its routes here, its first entry
   extern __shared__ int n_of[];
   int* base_of = n_of + hot_size + cold_size;
@@ -589,7 +588,7 @@
     // a padded token's routes are dropped, as if its ids were -1
     if (padding != nullptr && padding[r / TOPK]) e = -1;
     wt = topk_weights[r] * routed_scale;
-    if (e >= 0 && E > 0 && e >= E) e = -1;
+    if (e >= (E > 0 ? E : num_experts)) e = -1;
   }
   // without a placement the slot maps are read with the ids, not after them
   __shared__ int maps[2 * MAX_EXPERTS];
@@ -707,7 +706,15 @@
     ws->done_s = 0;
     ws->next[0] = ws->next[1] = 0;
     ws->T = num_tokens;
-    ws->epoch += 1;
+    // unsigned wrap; at 0 the ready flags are cleared so that a flag last
+    // set 2^32 calls ago cannot alias the new epoch
+    int next = static_cast<int>(static_cast<unsigned>(ws->epoch) + 1u);
+    if (next == 0) {
+      for (int i = 0; i < 2 * MAX_LIST; ++i) (&ws->ready[0][0])[i] = 0;
+      ws->ready_s = 0;
+      next = 1;
+    }
+    ws->epoch = next;
   }
 }
 
@@ -789,12 +796,6 @@
   }
   __syncwarp();
 }
-__device__ __forceinline__ void fence_acq_rel_gpu() {
-  asm volatile("fence.acq_rel.gpu;" ::: "memory");
-}
-#ifndef TD_FIN_FENCE
-  #define TD_FIN_FENCE fence_acq_rel_gpu
-#endif
 __device__ __forceinline__ void consumer_sync() {
   asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
 }
@@ -1117,8 +1118,8 @@
       }
       SchedEntry& e = sq[k];
       int gn = 0;
-      // a multi-chunk group is not claimed past: its successor is claimed
-      // once the producer has issued it
+      // no claim-ahead past a w13 or shared w2 group: its successor is
+      // claimed once the producer has issued all of its chunks
       const bool heavy =
           gi < (q ? len[1] : len[0]) &&
           group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1).kind !=
@@ -1255,7 +1256,8 @@
           __syncwarp();
           if (lane == 0) {
             desc[s] = make_int4(hdr, ei, t0, ntok);
-            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
+            mbar_expect_tx(&full[s],
+                           W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
             tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
             tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
           }
@@ -1273,16 +1275,17 @@
           __syncwarp();
           if (lane == 0) {
             desc[s] = make_int4(hdr, ei, t, ntok);
-            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
+            mbar_expect_tx(&full[s],
+                           W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
             tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
             tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
           }
           // weights are in flight; only the activation rows wait for the
           // entry's ready (every copying lane acquires for itself)
           if ((ei | q << 16) != last_ready) {
-           if (!pre_ready) {
-            while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
-           }
+            if (!pre_ready) {
+              while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
+            }
             fence_proxy_async();
             last_ready = ei | q << 16;
           }
@@ -1386,12 +1389,12 @@
       fs_[0] = sd_f[s_][2 * tq];
       fs_[1] = sd_f[s_][2 * tq + 1];
       if (r1) {
-        const unsigned char* xr = ring + static_cast<size_t>(s_) * STAGE_BYTES +
-                                  W_BYTES + S_BYTES +
-                                  ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
+        const unsigned char* xr =
+            ring + static_cast<size_t>(s_) * STAGE_BYTES + W_BYTES + S_BYTES +
+            ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
         fs_[0] *= *reinterpret_cast<const float*>(xr);
-        fs_[1] *= *reinterpret_cast<const float*>(
-            xr + (XROWS > 1 ? XROW_STRIDE : 0));
+        fs_[1] *=
+            *reinterpret_cast<const float*>(xr + (XROWS > 1 ? XROW_STRIDE : 0));
       }
 #pragma unroll
       for (int j = 0; j < MAX_TOK / 2; ++j)
@@ -1590,8 +1593,7 @@
                           void* stream_ptr) {
   TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..32 tokens");
   TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
-  TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS,
-             "1..512 experts");
+  TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS, "1..512 experts");
   Params p{};
   TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
              "hot tier maps");
@@ -1644,4 +1646,3 @@
     return -4;
   return 0;
 }
-
```
## adv_check.py diff (round 1 → round 2)
```diff
--- adv_check_v1.py	2026-10-10 14:25:52.397590000 +0200
+++ adv_check.py	2026-10-10 14:28:01.633713000 +0200
@@ -6,6 +6,12 @@
 shared-only, large / tiny / zero activations, odd token counts, workspace
 reused across M and routing changes, and a CUDA graph replaying several
 different calls back to back (PDL chain).
+Round 2 (Astra review 13): slot maps that permute the tiers, experts in
+neither tier, an empty hot or cold tier, the padding argument, all routes
+masked without the shared expert, concentrated cold routing, isolated routes
+(one-hot weights, no shared), per-row checks (exact zero where the reference
+row is zero), graph replays with new inputs every replay, and the ready epoch
+wrapping past 2^32 with stale flags that alias the next epoch.
   adv_check.py --v td_v53[:FLAGS] [--reps N] [--seed S]"""
 import argparse
 import json
@@ -29,9 +35,12 @@
     ck = T._int4_experts(E, gen)
     slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]
 
-    def split(tier):
-        hot = {k: v[:N_HOT].contiguous() for k, v in tier.items()}
-        cold = T._to_grace({k: v[N_HOT:].contiguous() for k, v in tier.items()}, dev)
+    def split(tier, ph=None, pc=None):
+        ph = torch.arange(N_HOT) if ph is None else ph
+        pc = torch.arange(N_COLD) if pc is None else pc
+        hot = {k: v[:N_HOT][ph.to(v.device)].contiguous() for k, v in tier.items()}
+        cold = T._to_grace({k: v[N_HOT:][pc.to(v.device)].contiguous()
+                            for k, v in tier.items()}, dev)
         return hot, cold
 
     sg = torch.Generator().manual_seed(7)
@@ -39,7 +48,10 @@
     sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)
     rows = lambda r: torch.cat([torch.arange(512 * r, 512 * r + 512),  # noqa: E731
                                 2048 + torch.arange(512 * r, 512 * r + 512)])
+    pg_ = torch.Generator().manual_seed(3)
+    ph, pc = torch.randperm(N_HOT, generator=pg_), torch.randperm(N_COLD, generator=pg_)
     return {"tiers": [split(sl) for sl in slices],
+            "ptiers": [split(sl, ph, pc) for sl in slices], "ph": ph, "pc": pc,
             "w13": T._int4_dequant(ck[0], ck[2]).to(dev),
             "w2": T._int4_dequant(ck[1], ck[3]).to(dev),
             "sw13": sw13.to(dev), "sw2": sw2.to(dev),
@@ -77,6 +89,12 @@
     elif kind == "one_hot":
         ids = torch.full((m, TOPK), -1)
         ids[0, TOPK - 1] = 17
+    elif kind == "same_cold":  # every token on the same 8 cold experts
+        ids = pick(allx[N_HOT:], TOPK)[:1].expand(m, TOPK).clone()
+    elif kind == "isolate":  # sparse ids; the weights pick one route per token
+        ids = pick(allx, TOPK)
+    elif kind == "dup_cold":  # every route on one cold expert (not top-k)
+        ids = torch.full((m, TOPK), N_HOT + 2)
     else:
         raise ValueError(kind)
     return ids.to(torch.int32)
@@ -95,28 +113,72 @@
     rscale = 2.5 if ver >= 28 else 1.0
     ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
     e = torch.empty(0, device=dev)
-    hm = torch.full((E,), -1, dtype=torch.int32, device=dev)
-    cm = torch.full((E,), -1, dtype=torch.int32, device=dev)
-    hm[:N_HOT] = torch.arange(N_HOT, dtype=torch.int32, device=dev)
-    cm[N_HOT:] = torch.arange(N_COLD, dtype=torch.int32, device=dev)
     kf = {"w13": f["w13"], "w2": f["w2"], "sw13": f["sw13"], "sw2": f["sw2"]}
 
+    def maps(hot_ids, cold_ids):
+        hm = torch.full((E,), -1, dtype=torch.int32)
+        cm = torch.full((E,), -1, dtype=torch.int32)
+        hm[hot_ids] = torch.arange(len(hot_ids), dtype=torch.int32)
+        cm[cold_ids] = torch.arange(len(cold_ids), dtype=torch.int32)
+        return hm.to(dev), cm.to(dev)
+
+    hot_all, cold_all = torch.arange(N_HOT), torch.arange(N_HOT, E)
+    absent = torch.tensor([3, 17, 45])
+    # name -> (tiers per slice, hot map, cold map, experts present here)
+    configs = {
+        "ident": (f["tiers"], *maps(hot_all, cold_all), torch.ones(E, dtype=torch.bool)),
+        "perm": (f["ptiers"], *maps(f["ph"], N_HOT + f["pc"]),
+                 torch.ones(E, dtype=torch.bool)),
+    }
+    hm, cm = maps(hot_all, cold_all)
+    hm[absent[absent < N_HOT]] = -1
+    cm[absent[absent >= N_HOT]] = -1
+    pres = torch.ones(E, dtype=torch.bool)
+    pres[absent] = False
+    configs["absent"] = (f["tiers"], hm, cm, pres)
+    empty = {k: e for k in f["tiers"][0][0]}
+    hm, cm = maps(hot_all, cold_all)
+    configs["nohot"] = ([(empty, c) for _, c in f["tiers"]], torch.full_like(hm, -1), cm,
+                        torch.arange(E) >= N_HOT)
+    configs["nocold"] = ([(h, empty) for h, _ in f["tiers"]], hm, torch.full_like(cm, -1),
+                         torch.arange(E) < N_HOT)
+
     def parts(t):
         return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                 t["w2_weight_scale"])
 
-    def call(out, x, ids, wt, r, shared=True):
-        hot, cold = f["tiers"][r]
+    def call(out, x, ids, wt, r, shared=True, cfg="ident", pad=None):
+        tiers, hm, cm, _ = configs[cfg]
+        hot, cold = tiers[r]
         extra = (*f["sslices"][r], SSCALE) if shared else (None, None, 1.0)
         mod.forward(out, x, ids, wt, hm, cm, e, e, e, 0, False, *parts(hot),
-                    *parts(cold), ws, True, e, *extra, rscale)
+                    *parts(cold), ws, True, e if pad is None else pad, *extra, rscale)
+
+    def ref_of(x, ids, wt, r, shared, cfg="ident", pad=None):
+        """The reference drops what the kernel must drop: experts absent here
+        and padded tokens' routes."""
+        keep = configs[cfg][3].to(dev)[ids.clamp_min(0).long()] & (ids >= 0)
+        if pad is not None:
+            keep &= ~pad[:, None]
+        return kdev.reference(kf, x, torch.where(keep, ids, -1), wt, shared, SSCALE, r,
+                              rscale)
 
     def err_of(out, ref):
-        return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
+        """Worst per-row error relative to the row's max |ref|; a zero
+        reference row must come out exactly zero."""
+        o = out.float()
+        if bool(torch.isnan(o).any()):
+            return float("nan")
+        rmax = ref.abs().amax(1)
+        d = (o - ref).abs().amax(1)
+        zero = rmax == 0
+        if bool((d[zero] != 0).any()):
+            return float("inf")
+        return float((d[~zero] / rmax[~zero]).max()) if bool((~zero).any()) else 0.0
 
     g = torch.Generator().manual_seed(a.seed)
     kinds = ["sparse", "same", "heavy", "cold", "hot", "masked", "shared_only",
-             "one_cold", "one_hot"]
+             "one_cold", "one_hot", "same_cold"]
     ms = [1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 24, 31, 32]
     scales = {"normal": 0.3, "large": 30.0, "tiny": 3e-4}
     worst, fails, n = 0.0, 0, 0
@@ -127,73 +189,151 @@
         for sc in ("large", "tiny"):
             cases.append((m, "sparse", sc, True))
         cases.append((m, "sparse", "zero_rows", True))
-        cases.append((m, "sparse", "normal", False))  # no shared expert
-    # interleave M and routing so the workspace sees every transition
+        for kind in ("sparse", "shared_only", "isolate", "one_cold", "one_hot", "masked"):
+            cases.append((m, kind, "normal", False))  # no shared expert
+    cfg_names = ["ident", "ident", "perm", "absent", "nohot", "nocold"]
+    by_cfg = {c: 0 for c in configs}
+
+    def inputs(m, kind, sc="normal"):
+        ids = routing(kind, m, g)
+        x = torch.randn((m, HIDDEN), generator=g) * scales.get(sc, 0.3)
+        if sc == "zero_rows":
+            x[::2] = 0
+        wt = torch.rand((m, TOPK), generator=g)
+        if kind == "isolate":
+            wt = torch.nn.functional.one_hot(torch.randint(0, TOPK, (m,), generator=g),
+                                             TOPK).float()
+        return x.to(torch.bfloat16).to(dev), ids.to(dev), wt.to(dev)
+
+    # interleave M, routing and map configs so the workspace sees every transition
     order = torch.randperm(len(cases), generator=g).tolist()
     for rep in range(a.reps):
         for ci in order:
             m, kind, sc, shared = cases[ci]
-            ids = routing(kind, m, g)
-            x = torch.randn((m, HIDDEN), generator=g) * scales.get(sc, 0.3)
-            if sc == "zero_rows":
-                x[::2] = 0
-            x = x.to(torch.bfloat16).to(dev)
-            wt = torch.rand((m, TOPK), generator=g).to(dev)
-            ids = ids.to(dev)
+            cfg = cfg_names[int(torch.randint(0, len(cfg_names), (1,), generator=g))]
+            x, ids, wt = inputs(m, kind, sc)
+            pad = None
+            if float(torch.rand(1, generator=g)) < 0.25:
+                pad = (torch.rand(m, generator=g) < 0.3).to(dev)
+            by_cfg[cfg] += 1
             for r in range(4):
-                ref = kdev.reference(kf, x, ids, wt, shared, SSCALE, r, rscale)
+                ref = ref_of(x, ids, wt, r, shared, cfg, pad)
                 out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
-                call(out, x, ids, wt, r, shared)
-                if kind == "shared_only" and not shared:
-                    continue
+                call(out, x, ids, wt, r, shared, cfg, pad)
                 err = err_of(out, ref)
-                bad = not (err <= 5e-3) or bool(torch.isnan(out).any())
-                if ref.abs().max() == 0:  # all routes masked, no shared: exact zero
-                    bad = bool(out.float().abs().max() != 0)
-                    err = float(out.float().abs().max())
                 n += 1
-                worst = max(worst, err if err == err else 1e9)
-                if bad:
+                worst = max(worst, err if err == err and err != float("inf") else 1e9)
+                if not (err <= 5e-3):
                     fails += 1
-                    print(json.dumps({"rep": rep, "m": m, "kind": kind, "x": sc,
-                                      "shared": shared, "slice": r, "err": err}), flush=True)
-    # CUDA graph: several different calls replayed back to back (PDL chain)
-    gcases = [(32, "sparse"), (1, "one_cold"), (8, "same"), (17, "masked"),
-              (32, "heavy"), (5, "cold")]
-    ins, outs, refs = [], [], []
-    for m, kind in gcases:
-        ids = routing(kind, m, g).to(dev)
-        x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16).to(dev)
-        wt = torch.rand((m, TOPK), generator=g).to(dev)
-        ins.append((x, ids, wt))
+                    print(json.dumps({"rep": rep, "m": m, "kind": kind, "x": sc, "cfg": cfg,
+                                      "pad": pad is not None, "shared": shared, "slice": r,
+                                      "err": err}), flush=True)
+    # every route of every token on one cold expert: outside top-k's contract
+    # (no repeated ids per token), reported but not part of the verdict
+    dup = []
+    for m in (1, 8, 32):
+        x, ids, wt = inputs(m, "dup_cold")
+        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
+        call(out, x, ids, wt, 0)
+        dup.append(err_of(out, ref_of(x, ids, wt, 0, True)))
+
+    # CUDA graph: different calls replayed back to back (PDL chain), with new
+    # inputs before every replay; shared on/off and an empty call in the chain
+    gcases = [(32, "sparse", True), (1, "one_cold", True), (8, "same", True),
+              (17, "masked", False), (4, "shared_only", False), (32, "heavy", True),
+              (5, "cold", False), (9, "isolate", False)]
+    bufs, outs = [], []
+    for m, kind, sh in gcases:
+        bufs.append(inputs(m, kind))
         outs.append(torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev))
-        refs.append(kdev.reference(kf, x, ids, wt, True, SSCALE, 0, rscale))
     s = torch.cuda.Stream()
     s.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(s):
-        for (x, ids, wt), o in zip(ins, outs):
-            call(o, x, ids, wt, 0)
+        for (x, ids, wt), o, (_, _, sh) in zip(bufs, outs, gcases):
+            call(o, x, ids, wt, 0, sh)
     torch.cuda.current_stream().wait_stream(s)
     graph = torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):
-        for (x, ids, wt), o in zip(ins, outs):
-            call(o, x, ids, wt, 0)
+        for (x, ids, wt), o, (_, _, sh) in zip(bufs, outs, gcases):
+            call(o, x, ids, wt, 0, sh)
     gfails = 0
-    for it in range(50 * a.reps):
-        for o in outs:
+    for it in range(20 * a.reps):
+        refs = []
+        for (x, ids, wt), o, (m, kind, sh) in zip(bufs, outs, gcases):
+            nx, nids, nwt = inputs(m, kind)
+            x.copy_(nx)
+            ids.copy_(nids)
+            wt.copy_(nwt)
             o.fill_(float("nan"))
+            refs.append(ref_of(x, ids, wt, 0, sh))
         graph.replay()
         torch.cuda.synchronize()
-        for (m, kind), o, ref in zip(gcases, outs, refs):
+        for (m, kind, sh), o, ref in zip(gcases, outs, refs):
             err = err_of(o, ref)
             n += 1
-            worst = max(worst, err if err == err else 1e9)
+            worst = max(worst, err if err == err and err != float("inf") else 1e9)
             if not (err <= 5e-3):
                 gfails += 1
-                print(json.dumps({"graph_iter": it, "m": m, "kind": kind, "err": err}), flush=True)
+                print(json.dumps({"graph_iter": it, "m": m, "kind": kind, "err": err}),
+                      flush=True)
     fails += gfails
-    print(json.dumps({"v": a.v, "calls_checked": n, "worst_rel_err": round(worst, 5),
-                      "fails": fails, "graph_fails": gfails, "pass": fails == 0}))
+
+    # epoch wrap: find the epoch word (the only one an empty call bumps) and
+    # the ready flags (bumped by a repeated busy call), then set the epoch to
+    # 2^32 - 1 and every flag to 0, the epoch a plain increment would wrap to
+    w32 = ws.view(torch.int32)
+    xa, ia, wa = inputs(32, "sparse")
+    out = torch.empty((32, HIDDEN), dtype=torch.bfloat16, device=dev)
+    call(out, xa, ia, wa, 0)
+    w0 = w32.clone()
+    call(out, xa, ia, wa, 0)
+    w1 = w32.clone()
+    xe, ie, we = inputs(4, "shared_only")
+    call(torch.empty((4, HIDDEN), dtype=torch.bfloat16, device=dev), xe, ie, we, 0, False)
+    w2 = w32.clone()
+    ep = (w2 == w1 + 1).nonzero().flatten()
+    assert len(ep) == 1 and int(w1[ep[0]]) == int(w0[ep[0]]) + 1, len(ep)
+    # ready flags: bumped by the repeated call and equal to its epoch
+    flags = ((w1 == w0 + 1) & (w1 == w1[ep[0]])).nonzero().flatten()
+    wfails = 0
+    for trial in range(10):
+        w32[flags] = 0
+        w32[ep] = -1
+        for i in range(3):  # the wrapping call, then two ordinary ones
+            x, ids, wt = inputs(32, "sparse")
+            out = torch.full((32, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
+            call(out, x, ids, wt, 0)
+            err = err_of(out, ref_of(x, ids, wt, 0, True))
+            n += 1
+            if not (err <= 5e-3):
+                wfails += 1
+                print(json.dumps({"wrap_trial": trial, "call": i, "err": err,
+                                  "epoch": int(w32[ep[0]])}), flush=True)
+        # the wrapping call must skip epoch 0
+        if ver >= 55 and int(w32[ep[0]]) != 3:
+            wfails += 1
+            print(json.dumps({"wrap_trial": trial, "epoch_after": int(w32[ep[0]])}))
+    fails += wfails
+    # ids past the maps (num_experts .. 511, and past MAX_EXPERTS) are dropped
+    ofails = 0
+    if ver >= 55:
+        for m in (1, 8, 32):
+            x, ids, wt = inputs(m, "sparse")
+            bad = torch.rand(ids.shape, generator=g).to(dev) < 0.3
+            junk = torch.where(torch.rand(ids.shape, generator=g).to(dev) < 0.5, E + 3, 700)
+            ids_bad = torch.where(bad, junk.to(ids.dtype), ids)
+            out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
+            call(out, x, ids_bad, wt, 0)
+            err = err_of(out, ref_of(x, torch.where(bad, -1, ids), wt, 0, True))
+            n += 1
+            if not (err <= 5e-3):
+                ofails += 1
+                print(json.dumps({"oob_m": m, "err": err}), flush=True)
+    fails += ofails
+    print(json.dumps({"v": a.v, "oob_fails": ofails, "calls_checked": n, "worst_rel_err": round(worst, 5),
+                      "fails": fails, "graph_fails": gfails, "wrap_fails": wfails,
+                      "cases_by_cfg": by_cfg,
+                      "dup_cold_err": [round(d, 5) for d in dup], "pass": fails == 0}))
     sys.exit(0 if fails == 0 else 1)
 
 
```
