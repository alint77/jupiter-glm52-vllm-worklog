# TP-sliced tiered MoE in vLLM: integration plan (2026-10-09)

Goal: serve GLM-5.3 W4A16 with every GPU holding a 512-wide slice of every
routed expert (hot slices in HBM, cold in its own Grace), the decode MoE on the
sliced persistent kernel (td_v27, shared expert fused), and A/B it served
against prod EP (offload + replicas + balancer) on the agentic task set.

## Approach: TP MoE (EP off) + the existing tiered machinery

vLLM with TP=4 and `enable_expert_parallel=False` already gives each rank every
expert at `intermediate_size_per_partition = 512`. The shared expert is TP-sliced the same way,
and the post-MoE all-reduce sums the partials. Everything EP-only is turned
off in this mode: replicas, the balancer and cold prefetch. The tiered
planner/storage/streaming are reused with "every rank owns all 256 experts at
slice size".

1. **Config** (`vllm/config/tiered_moe.py`): `layout: Literal["ep",
   "tp_sliced"] = "ep"`. `tp_sliced` requires EP off, `replica_assignment ==
   "off"`, cold prefetch off, and the wgmma prefill kernel off. The current
   validator, which requires EP, applies only to `ep`.
2. **Manifest / storage layout**: slice component specs
   (`glm_marlin_components` for N = 2 x 512, K = 512 on w2; scales likewise).
   `runtime_expert_bytes` becomes slice bytes (1/4) in this mode; checkpoint
   bytes are unchanged, since every rank reads whole expert tensors and slices
   them. That's 4x load I/O; partial safetensors reads can fix it later.
3. **Physical plan** (`tiered_moe_physical.build_tiered_moe_rank_load_plan`,
   `plan_tiered_moe_scenario`): in `tp_sliced`, plan one rank with
   `ep_size=1` (owned = all experts per layer). The hot map is the union over
   ranks of the EP profile's `hot_for_rank`; the planner promotes/demotes it to
   the slot count from available HBM / slice bytes. The same plan applies to
   all TP ranks, so every rank has the same hot set. Non-routed inventory stays
   TP4. Runtime buffers: lift the EP4-only check; size Marlin arenas for
   inter 512, 256 local experts.
4. **Conversion** (`convert_glm_w4a16_expert`): in `tp_sliced`, before
   transposition take rank r's slice of the packed checkpoint tensors:
   - gate/up `weight_packed` rows [512r, 512r+512) and `weight_scale` rows
     likewise;
   - down `weight_packed` columns [64r, 64r+64) (512 K / 8 per int32) and
     `weight_scale` columns [16r, 16r+16);
   - shapes (6144, 512) / (512, 6144), g_idx 6144 / 512.

   The Marlin repack then uses the TP layer (512). The bench's `slice_ckpt` +
   `_int4_marlin_tier` did exactly this on unpacked codes and matched fp32
   (2.2e-3).
5. **Decode**: `apply_tiered_moe` in `tp_sliced` with T <= 32 calls a new
   wrapper (`tiered_decode/sliced.py`). It nvcc-builds `sliced_decode.cu`
   (td_v27, torch-free C ABI) once into VLLM_CACHE_ROOT (per-rank tmp +
   atomic rename) and calls `td_forward` through ctypes on the current stream
   (graph-capturable: host launch only). Inputs are the hot/cold Marlin
   component tensors, the static hot/cold slot maps, a persistent workspace
   tensor, and `pdl_launch`.
   - Phase A: shared expert NOT fused (left to the runner).
   - Phase B: fused. Pass the shared expert's gate_up_proj [1024, 6144] and
     down_proj [6144, 512] bf16 weights and `shared_scale =
     1/routed_scaling_factor`, and make the runner skip its shared call on
     these steps. The kernel returns routed + shared / rsf, so the runner's
     `* rsf` gives routed * rsf + shared.
6. **Prefill / large steps**: unchanged Marlin hot + cold tier calls with
   static expert maps (cold slices read from Grace through UVA), at 512 inter.
7. **Validation**:
   - (a) a unit test of the wrapper against fp32 on random experts (kdev check
     cases);
   - (b) server boots, greedy outputs coherent, and an eval (gsm8k-like or the
     agentic acceptance) on par with EP;
   - (c) served A/B on the agentic task set vs prod EP (same serve.sh
     otherwise) at c=1/2/4/8, with step-time profiles (breakdown2.py).

## Risks / open questions
- Load time ~4x the I/O per rank (all expert bytes read per rank).
- HBM budget: Marlin runtime arenas sized for 2048 / 64 local experts today.
- TP MoE path with EP off: does the FusedMoE / modular Marlin path (prefill)
  accept a tiered layer with 256 local experts and expert maps?
- DCP4 / MTP / DFlash2 paths assume EP anywhere?
- The 75 MoE layers' all-reduce is the same in both modes (partial sums).
