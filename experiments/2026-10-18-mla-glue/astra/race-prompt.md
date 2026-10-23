# Review request: find a race / lifetime / ordering bug in a fused MLA "glue" kernel (vLLM, GLM-5.3, GH200)

You cannot run commands (the sandbox fails on this cluster). Reason from the
code and evidence below only. Do not run tools.

## Task

A fused CUDA kernel (`mla_qkv_glue`) replaces MLA's work between
`fused_qkv_a_proj` and `q_b_proj` in decode: q_a RMSNorm (2048), kv_a RMSNorm
(512), k_pe interleaved RoPE, and the fp8_ds_mla KV-cache writes (main cache
and the skip-KV stager's reserved rows on index-share layers). It is
registered as a torch custom op that allocates its own outputs (q_c,
kv_c_normed, k_pe), runs inside vLLM's torch.compile pieces and FULL CUDA
graphs (decode), and uses PDL (programmatic dependent launch).

Op-level tests pass (k_pe bitwise vs fp32 ref; norms within 1 ulp; cache bytes
equal concat_and_cache_mla of its own outputs). Prefill is clean. But every
served glue server **decodes worse**: MTP acceptance drops (prod 3.43 -> glue
1.64-2.03) and decode logprobs get less confident. Evidence below.

Find the bug in the code. Rank hypotheses by how well they explain ALL the
evidence (especially: replacing any ONE of glue's three outputs with a
recomputed tensor restores prod). For each: the exact mechanism, file:line,
why it fits / doesn't fit each row of the evidence, and the cheapest
discriminating probe (login GPU = correctness only; Booster node available
for served runs) and the fix. Then tell us what to do first.

## Evidence (worklog README, verbatim)

## Decode quality regression (open)

MTP acceptance on the dec1 arms: prod 3.43 / 3.14, run-2 b 2.01 / 2.03, run-3
c 1.64. pf6 and pf7 differ only by glue.diff. Op checks on the login GPU
(correctness only): kernel vs vLLM's own `RMSNorm.forward_native` and
`RotaryEmbedding.forward_cuda` (k_pe bitwise, norms 1 ulp); q RoPE with
`key=None` on the strided slice = with k. PDL overlap is real (`q_b`'s
decode_gemm starts before glue ends in every step; decode_gemm loads weights
before `griddepcontrol.wait`, activations after; its own `launch_dependents`
precedes its y stores, which the dependent's wait covers).

Bisection (`launch_probe.sh`, `probe.py`, `probe_arm.sh`; one server per
hold at the c=4 config; pf9 = pf8 + `debug-knobs-pf9.diff`; probe = prompt
logprobs of a 3K prompt, greedy 4 (mgp) / 8 (mgq) x 5K chats of 300 tokens,
sampled 4 x 5K; `probe_summary.txt`):

| variant | servers | greedy lp | sampled lp |
|---|--:|---|---|
| prod (pf6) | 4 | -0.123 .. -0.128 | -0.236 .. -0.290 |
| glue (pf9) | 4 | **-0.171 .. -0.210** | **-0.546 .. -0.600** |
| krope: glue, downstream k_pe from rotary_emb | 3 | -0.116 .. -0.137 | -0.237 .. -0.318 |
| kvorig: + kv_c_normed from the module | 1 | -0.124 | -0.246 |
| qorig: q_c from the module, glue's kv outputs | 1 | -0.130 | -0.308 |
| nopdl / nodraft / both | 1 each | -0.177 / -0.232 / -0.172 | -0.312 / -0.387 / -0.277 |

- Prefill is clean: prompt-logprob mean |d| glue vs prod 0.061-0.078 = prod
  vs prod 0.065 (serving is not bitwise deterministic; greedy outputs of two
  prod servers diverge within 2 tokens too).
- Decode is not: every full-glue server is less confident, by far the most
  under sampling.
- Not the math: krope changes only the k_pe *tensor* glue hands on, and
  nothing reads it in a decode step (MQA attention reads q and the cache; the
  cache update is skipped; glue writes the cache from its in-kernel values in
  both). qorig changes only q_c. Each alone restores prod. Neither PDL nor
  the MTP layer accounts for it (nopdl / nodraft stay bad).
- So likely ordering / lifetime: something reads glue's outputs (or what it
  writes) while it is still changing, and any extra kernel in between narrows
  the window. Leads: the custom op allocates qn / kvn / kpe itself (inside the
  compiled pieces and the FULL capture), so their reuse is the caching
  allocator's, not inductor's; the side streams in the layer (cold-prefetch
  fork/join, skip-KV stager, L2 prefetch). Next probes: `--enforce-eager`
  glue vs prod; glue with outputs passed in from the caller (inductor-owned);
  compute-sanitizer racecheck on one replayed step.

The perf numbers above (run 2) measure a path that decodes worse; they are
not a ship case.


Design notes from the same README:
3. **`mla_glue.writes_kv()`** = stream capturing and attention metadata
   present (FULL capture only: piecewise captures have none, eager steps and
   piecewise replays are not capturing). The op and the kv-cache-update skip
   both use it, so a forward never writes twice or not at all. Run mg3
   (`pf8`, `launch_ab3.sh`).


## The glue change (glue.diff, vs prod tree pf6)
```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index 20e05babcf..eb5ba1933e 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -282,6 +282,7 @@ if TYPE_CHECKING:
     VLLM_DCP_ONE_SHOT_COLLECTIVES: bool = False
     VLLM_DCP_SPARSE_DECODE_WIDTH: int = 0
     VLLM_DCP_DECODE_SCHED: bool = True
+    VLLM_MLA_QKV_GLUE: bool = False
     VLLM_SCHED_INTERLEAVE_PREFILL: int = 0
     VLLM_SCHED_INTERLEAVE_SRPT: bool = False
     VLLM_TIERED_MOE_KV_POOL_SEQS: int = 0
@@ -2021,6 +2022,10 @@ environment_variables: dict[str, Callable[[], Any]] = {
     # the longest live row and splitting them evenly, instead of FlashMLA's
     # planner (full width, unbalanced splits for a single request).
     "VLLM_DCP_DECODE_SCHED": lambda: bool(int(os.getenv("VLLM_DCP_DECODE_SCHED", "1"))),
+    # MLA (GLM-5.3 layout, fp8_ds_mla cache, sm90): the q_a / kv_a RMSNorms,
+    # the k_pe RoPE and the layer's KV writes between fused_qkv_a and q_b in
+    # one kernel (vllm/model_executor/layers/mla_glue).
+    "VLLM_MLA_QKV_GLUE": lambda: bool(int(os.getenv("VLLM_MLA_QKV_GLUE", "0"))),
     # Scheduler: never mix prefill and decode tokens in one step. With N > 0,
     # a step is either decode-only or one request's prefill chunk, and N
     # decode-only steps run between prefill chunks while requests decode.
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index c394cc7a88..141d466276 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -613,14 +613,15 @@ class MLAAttention(nn.Module, AttentionLayerBase):
             assert isinstance(slot_mapping, dict), (
                 f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
             )
-            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
-                kv_c_normed,
-                k_pe,
-                self_kv_cache,
-                slot_mapping.get(self.layer_name),
-                self.kv_cache_dtype,
-                self._k_scale,
-            )
+            if not _glue_wrote_kv(self):
+                self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
+                    kv_c_normed,
+                    k_pe,
+                    self_kv_cache,
+                    slot_mapping.get(self.layer_name),
+                    self.kv_cache_dtype,
+                    self._k_scale,
+                )
             output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
             self.forward_impl(
                 q,
@@ -1106,6 +1107,13 @@ class MLAAttention(nn.Module, AttentionLayerBase):
             torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
 
 
+def _glue_wrote_kv(layer: "MLAAttention") -> bool:
+    """vllm::mla_qkv_glue wrote the layer's KV itself (FULL graph capture)."""
+    from vllm.model_executor.layers import mla_glue
+
+    return getattr(layer, "qkv_glue", False) and mla_glue.writes_kv()
+
+
 def unified_mla_kv_cache_update(
     kv_c_normed: torch.Tensor,
     k_pe: torch.Tensor,
@@ -1119,7 +1127,7 @@ def unified_mla_kv_cache_update(
     """
     layer_name = _resolve_layer_name(layer_name)
     _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
-    if layer_slot_mapping is not None:
+    if layer_slot_mapping is not None and not _glue_wrote_kv(attn_layer):
         attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
             kv_c_normed,
             k_pe,
diff --git a/vllm/model_executor/layers/mla.py b/vllm/model_executor/layers/mla.py
index 2bcf9127e0..c03dd16ba0 100644
--- a/vllm/model_executor/layers/mla.py
+++ b/vllm/model_executor/layers/mla.py
@@ -8,6 +8,7 @@ from vllm.config import CacheConfig
 from vllm.model_executor.custom_op import PluggableLayer
 from vllm.model_executor.layers.attention import MLAAttention
 from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.utils.torch_utils import _encode_layer_name
 
 
 @dataclass
@@ -124,6 +125,11 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
 
         self.cold_prefetch_layer = cold_prefetch_layer(prefix)
 
+        from vllm.model_executor.layers import mla_glue
+
+        self.qkv_glue = mla_glue.supported(self)
+        self.mla_attn.qkv_glue = self.qkv_glue
+
     def forward(
         self,
         positions: torch.Tensor,
@@ -133,7 +139,25 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
         q_c = None
         kv_lora = None
 
-        if self.q_lora_rank is not None:
+        if self.qkv_glue:
+            assert self.fused_qkv_a_proj is not None
+            assert self.q_a_layernorm is not None and self.q_b_proj is not None
+            q_c, kv_c_normed, k_pe = torch.ops.vllm.mla_qkv_glue(
+                    self.fused_qkv_a_proj(hidden_states)[0],
+                    positions,
+                    self.rotary_emb.cos_sin_cache,
+                    self.q_a_layernorm.weight,
+                    self.kv_a_layernorm.weight,
+                    self.q_a_layernorm.variance_epsilon,
+                    self.kv_a_layernorm.variance_epsilon,
+                    _encode_layer_name(self.mla_attn.layer_name),
+            )
+            q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
+            k_pe = k_pe.unsqueeze(1)
+            q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
+                positions, q[..., self.qk_nope_head_dim :]
+            )
+        elif self.q_lora_rank is not None:
             assert self.fused_qkv_a_proj is not None, (
                 "fused_qkv_a_proj is required when q_lora_rank is not None"
             )
@@ -161,17 +185,20 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
             kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
             q = self.q_proj(hidden_states)[0]
 
-        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
-        kv_c_normed = self.kv_a_layernorm(kv_c)
+        if not self.qkv_glue:
+            kv_c, k_pe = kv_lora.split(
+                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
+            )
+            kv_c_normed = self.kv_a_layernorm(kv_c)
 
-        q = q.view(-1, self.num_heads, self.qk_head_dim)
-        # Add head dim of 1 to k_pe
-        k_pe = k_pe.unsqueeze(1)
+            q = q.view(-1, self.num_heads, self.qk_head_dim)
+            # Add head dim of 1 to k_pe
+            k_pe = k_pe.unsqueeze(1)
 
-        if self.rotary_emb is not None:
-            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
-                positions, q[..., self.qk_nope_head_dim :], k_pe
-            )
+            if self.rotary_emb is not None:
+                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
+                    positions, q[..., self.qk_nope_head_dim :], k_pe
+                )
 
         if self.cold_prefetch_layer is not None:
             torch.ops.vllm.tiered_moe_cold_join(q)
diff --git a/vllm/model_executor/layers/mla_glue/__init__.py b/vllm/model_executor/layers/mla_glue/__init__.py
new file mode 100644
index 0000000000..3e5d0564a3
--- /dev/null
+++ b/vllm/model_executor/layers/mla_glue/__init__.py
@@ -0,0 +1,173 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MLA's work between fused_qkv_a and q_b in one kernel (mla_glue.cu): the
+q_a / kv_a RMSNorms, the k_pe RoPE and the layer's fp8_ds_mla KV writes
+(main cache, and the skip-KV stage's reserved rows on index-share layers).
+Registered as vllm::mla_qkv_glue. It writes the KV only while a FULL CUDA
+graph is captured (writes_kv), whose replays reuse the bound cache and slot
+buffers; unified_mla_kv_cache_update then skips the layer. In piecewise
+graphs and eager steps it only normalizes and rotates, and the splitting
+kv-cache op writes as before. VLLM_MLA_QKV_GLUE=1; built on first use for
+sm_90a, cached under VLLM_CACHE_ROOT."""
+
+from functools import cache
+from pathlib import Path
+
+import torch
+
+import vllm.envs as envs
+from vllm.forward_context import get_forward_context
+from vllm.utils.torch_utils import (
+    LayerNameType,
+    _resolve_layer_name,
+    direct_register_custom_op,
+)
+
+_HERE = Path(__file__).resolve().parent
+Q_LORA, KV_LORA, ROPE = 2048, 512, 64
+
+
+@cache
+def _extension():
+    from torch.utils.cpp_extension import load
+
+    build = Path(envs.VLLM_CACHE_ROOT) / "torch_extensions" / "mla_glue"
+    build.mkdir(parents=True, exist_ok=True)
+    return load(
+        name="vllm_mla_glue",
+        sources=[str(_HERE / "mla_glue.cu")],
+        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
+        build_directory=str(build),
+    )
+
+
+def supported(wrapper: torch.nn.Module) -> bool:
+    """Whether a MultiHeadLatentAttentionWrapper has the layout and ops the
+    kernel implements."""
+    from vllm.model_executor.layers.layernorm import RMSNorm
+    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
+    from vllm.platforms import current_platform
+    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
+
+    rope = wrapper.rotary_emb
+    norms = (wrapper.q_a_layernorm, wrapper.kv_a_layernorm)
+    return (
+        envs.VLLM_MLA_QKV_GLUE
+        and current_platform.is_device_capability(90)
+        # the cache writes it replaces (main cache + skip-KV reserved rows)
+        and type(wrapper.mla_attn.impl) is FlashMLASparseImpl
+        and wrapper.q_lora_rank == Q_LORA
+        and wrapper.kv_lora_rank == KV_LORA
+        and wrapper.qk_rope_head_dim == ROPE
+        and type(rope) is RotaryEmbedding
+        and not rope.is_neox_style
+        and rope.rotary_dim == rope.head_size == ROPE
+        and not rope.use_flashinfer
+        and rope.cos_sin_cache.dtype == torch.bfloat16
+        and all(
+            type(n) is RMSNorm
+            and n.has_weight
+            and n.variance_size_override is None
+            and n.weight.dtype == torch.bfloat16
+            for n in norms
+        )
+        and wrapper.mla_attn.kv_cache_dtype == "fp8_ds_mla"
+    )
+
+
+def writes_kv() -> bool:
+    """Whether the glue writes this forward's KV: inside a FULL graph capture,
+    which the V2 runner runs as a NONE-mode forward with attention metadata
+    (piecewise captures have none, and their replays run the kv-cache op)."""
+    fc = get_forward_context()
+    return (
+        torch.cuda.is_current_stream_capturing()
+        and fc.attn_metadata is not None
+        and bool(fc.slot_mapping)
+    )
+
+
+def _caches(
+    layer_name: str, num_tokens: int
+) -> tuple[torch.Tensor | None, ...]:
+    """(main cache, its slots, staged buffer, its rows): what
+    unified_mla_kv_cache_update and the skip-KV write_new would write."""
+    from vllm.model_executor.layers.attention.attention import (
+        get_attention_context,
+    )
+    from vllm.v1.attention.backends.mla import skip_kv_stage
+
+    _, layer, kv_cache, slots = get_attention_context(layer_name)
+    if slots is None or kv_cache.numel() == 0:
+        return None, None, None, None
+    slots = slots.flatten()
+    impl = layer.impl
+    if (
+        getattr(impl, "_skip_kv", False)
+        and impl._reuses_topk
+        and slots.numel() <= skip_kv_stage.limits()[0]
+    ):
+        stager = skip_kv_stage.get_stager(kv_cache.device)
+        p = stager.position.get(kv_cache.data_ptr())
+        if p is not None:
+            return (
+                kv_cache,
+                slots,
+                stager.buffers[p],
+                stager.reserved_map[: slots.numel()],
+            )
+    return kv_cache, slots, None, None
+
+
+def _impl(
+    qkv: torch.Tensor,
+    positions: torch.Tensor,
+    cos_sin_cache: torch.Tensor,
+    q_weight: torch.Tensor,
+    kv_weight: torch.Tensor,
+    q_eps: float,
+    kv_eps: float,
+    layer_name: LayerNameType,
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+    m = qkv.shape[0]
+    qn = qkv.new_empty(m, Q_LORA)
+    kvn = qkv.new_empty(m, KV_LORA)
+    kpe = qkv.new_empty(m, ROPE)
+    caches = (
+        _caches(_resolve_layer_name(layer_name), m) if writes_kv() else (None,) * 4
+    )
+    _extension().glue(
+        qkv,
+        positions,
+        cos_sin_cache,
+        q_weight,
+        kv_weight,
+        qn,
+        kvn,
+        kpe,
+        *caches,
+        q_eps,
+        kv_eps,
+    )
+    return qn, kvn, kpe
+
+
+def _fake(
+    qkv: torch.Tensor,
+    positions: torch.Tensor,
+    cos_sin_cache: torch.Tensor,
+    q_weight: torch.Tensor,
+    kv_weight: torch.Tensor,
+    q_eps: float,
+    kv_eps: float,
+    layer_name: LayerNameType,
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+    m = qkv.shape[0]
+    return (
+        qkv.new_empty(m, Q_LORA),
+        qkv.new_empty(m, KV_LORA),
+        qkv.new_empty(m, ROPE),
+    )
+
+
+direct_register_custom_op(op_name="mla_qkv_glue", op_func=_impl, fake_impl=_fake)
diff --git a/vllm/model_executor/layers/mla_glue/mla_glue.cu b/vllm/model_executor/layers/mla_glue/mla_glue.cu
new file mode 100644
index 0000000000..e9b258c55c
--- /dev/null
+++ b/vllm/model_executor/layers/mla_glue/mla_glue.cu
@@ -0,0 +1,225 @@
+// The glue between MLA's fused_qkv_a and q_b for the GLM-5.3 / DeepSeek-V3.2
+// layout, qkv[M, 2624] = [q_c 2048 | kv_c 512 | k_pe 64], in one kernel: the
+// q_a and kv_a RMSNorms, the interleaved RoPE of k_pe, and the layer's
+// fp8_ds_mla KV entry (concat_and_cache_ds_mla_kernel's: 512 e4m3 bytes, 4 fp32
+// tile scales, 64 bf16 RoPE values) into its cache and, for index-share layers,
+// into the skip-KV stage's reserved rows. One CTA per token. It lets its
+// dependent (q_b) launch at once, so q_b's first weight loads overlap it, and
+// waits for qkv_a before reading.
+//
+// Arithmetic as inductor's kernels for the served graph: fp32 sum of squares,
+// / n, + eps, rsqrt, x * r * w in fp32 with one bf16 rounding; RoPE in fp32
+// from bf16 inputs, one rounding. Only the sum order differs (one bf16 ulp on
+// rare elements). The KV entry is built from the bf16 kv_c and k_pe exactly
+// as concat_and_cache_ds_mla_kernel builds it.
+#include <ATen/cuda/CUDAContext.h>
+#include <c10/cuda/CUDAGuard.h>
+#include <cuda_bf16.h>
+#include <cuda_fp8.h>
+#include <torch/extension.h>
+
+#include <cfloat>
+
+namespace {
+
+constexpr int QL = 2048, KVL = 512, PE = 64, LD = QL + KVL + PE;
+constexpr int ENTRY = KVL + 16 + 2 * PE;  // 656 bytes
+constexpr int THREADS = 256;
+
+struct Cache {
+  uint8_t* base;
+  const int64_t* slots;
+  int64_t block_stride, entry_stride, num_slots;
+  int block_size;
+};
+
+template <int N>
+__device__ __forceinline__ float block_sum(float v, float* sh) {
+  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
+  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
+  __syncthreads();
+  float s = 0.f;
+#pragma unroll
+  for (int w = 0; w < N; ++w) s += sh[w];
+  return s;
+}
+
+__device__ __forceinline__ uint8_t* entry(const Cache& c, int m) {
+  if (c.base == nullptr || m >= c.num_slots) return nullptr;
+  const int64_t s = c.slots[m];
+  if (s < 0) return nullptr;
+  return c.base + (s / c.block_size) * c.block_stride +
+         (s % c.block_size) * c.entry_stride;
+}
+
+__global__ void __launch_bounds__(THREADS)
+    qkv_glue_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t ld,
+                    const int64_t* __restrict__ pos,
+                    const __nv_bfloat16* __restrict__ cos_sin, int64_t max_pos,
+                    const __nv_bfloat16* __restrict__ wq,
+                    const __nv_bfloat16* __restrict__ wkv,
+                    __nv_bfloat16* __restrict__ qn,
+                    __nv_bfloat16* __restrict__ kvn,
+                    __nv_bfloat16* __restrict__ kpe, Cache c1, Cache c2,
+                    float eps_q, float eps_kv) {
+  __shared__ float shq[THREADS / 32], shk[THREADS / 32], tmax[THREADS / 32];
+  asm volatile("griddepcontrol.launch_dependents;");
+  asm volatile("griddepcontrol.wait;" ::: "memory");
+  const int m = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
+  const __nv_bfloat16* row = qkv + m * ld;
+
+  // q_c: 8 elements per thread, kv_c: 2
+  const uint4 qv = reinterpret_cast<const uint4*>(row)[t];
+  const __nv_bfloat162 kv2 =
+      reinterpret_cast<const __nv_bfloat162*>(row + QL)[t];
+  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
+  float q[8];
+#pragma unroll
+  for (int i = 0; i < 4; ++i) {
+    q[2 * i] = __low2float(q2[i]);
+    q[2 * i + 1] = __high2float(q2[i]);
+  }
+  const float k0 = __low2float(kv2), k1 = __high2float(kv2);
+  float sq = 0.f;
+#pragma unroll
+  for (int i = 0; i < 8; ++i) sq += q[i] * q[i];
+  const float rq =
+      rsqrtf(block_sum<THREADS / 32>(sq, shq) / float(QL) + eps_q);
+  const float rk = rsqrtf(
+      block_sum<THREADS / 32>(k0 * k0 + k1 * k1, shk) / float(KVL) + eps_kv);
+
+  const uint4 wqv = reinterpret_cast<const uint4*>(wq)[t];
+  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wqv);
+  uint4 out;
+  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
+#pragma unroll
+  for (int i = 0; i < 4; ++i)
+    o2[i] = __floats2bfloat162_rn(q[2 * i] * rq * __low2float(w2[i]),
+                                  q[2 * i + 1] * rq * __high2float(w2[i]));
+  reinterpret_cast<uint4*>(qn + int64_t(m) * QL)[t] = out;
+
+  const __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(wkv)[t];
+  const __nv_bfloat162 kn = __floats2bfloat162_rn(
+      k0 * rk * __low2float(wk), k1 * rk * __high2float(wk));
+  reinterpret_cast<__nv_bfloat162*>(kvn + int64_t(m) * KVL)[t] = kn;
+
+  // k_pe pairs (2t, 2t + 1), cos / sin halves of the cache row; a negative
+  // position wraps as the served index_select lowering does
+  uint32_t pe = 0;
+  if (t < PE / 2) {
+    int64_t p = pos[m];
+    if (p < 0) p += max_pos;
+    const __nv_bfloat16* cs = cos_sin + p * PE;
+    const float c = __bfloat162float(cs[t]), s = __bfloat162float(cs[PE / 2 + t]);
+    const float x0 = __bfloat162float(row[QL + KVL + 2 * t]);
+    const float x1 = __bfloat162float(row[QL + KVL + 2 * t + 1]);
+    const __nv_bfloat162 r =
+        __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
+    reinterpret_cast<__nv_bfloat162*>(kpe + int64_t(m) * PE)[t] = r;
+    pe = *reinterpret_cast<const uint32_t*>(&r);
+  }
+
+  // KV entry: tile = 128 elements = 64 threads = 2 warps
+  float mx = fmaxf(fabsf(__low2float(kn)), fabsf(__high2float(kn)));
+  for (int o = 16; o; o >>= 1)
+    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
+  if (lane == 0) tmax[warp] = mx;
+  __syncthreads();
+  const int tile = t / 64;
+  const float scale =
+      fmaxf(fmaxf(tmax[2 * tile], tmax[2 * tile + 1]) / 448.f, FLT_MIN);
+  const uint16_t fp8 =
+      uint16_t(__nv_cvt_float_to_fp8(__low2float(kn) / scale, __NV_SATFINITE,
+                                     __NV_E4M3)) |
+      uint16_t(__nv_cvt_float_to_fp8(__high2float(kn) / scale, __NV_SATFINITE,
+                                     __NV_E4M3))
+          << 8;
+#pragma unroll
+  for (int k = 0; k < 2; ++k) {
+    uint8_t* e = entry(k ? c2 : c1, m);
+    if (e == nullptr) continue;
+    reinterpret_cast<uint16_t*>(e)[t] = fp8;
+    if (t % 64 == 0) reinterpret_cast<float*>(e + KVL)[tile] = scale;
+    if (t < PE / 2) reinterpret_cast<uint32_t*>(e + KVL + 16)[t] = pe;
+  }
+}
+
+Cache cache_of(const std::optional<torch::Tensor>& cache,
+               const std::optional<torch::Tensor>& slots,
+               const torch::Device& device) {
+  if (!cache.has_value()) return {nullptr, nullptr, 0, 0, 0, 1};
+  const auto& c = *cache;
+  TORCH_CHECK(slots.has_value(), "a KV cache needs its slot mapping");
+  const auto& s = *slots;
+  TORCH_CHECK(c.device() == device && s.device() == device);
+  TORCH_CHECK(c.dim() == 3 && c.size(2) == ENTRY &&
+                  c.scalar_type() == torch::kUInt8 && c.stride(2) == 1 &&
+                  c.size(1) > 0,
+              "expected an fp8_ds_mla cache [blocks, block_size, 656]");
+  TORCH_CHECK(c.stride(0) % 4 == 0 && c.stride(1) % 4 == 0 &&
+                  reinterpret_cast<uintptr_t>(c.data_ptr()) % 4 == 0,
+              "fp8_ds_mla cache entries must be 4-byte aligned");
+  TORCH_CHECK(s.dim() == 1 && s.scalar_type() == torch::kInt64 &&
+              s.stride(0) == 1);
+  return {c.data_ptr<uint8_t>(), s.data_ptr<int64_t>(), c.stride(0),
+          c.stride(1), s.size(0), static_cast<int>(c.size(1))};
+}
+
+void check_bf16(const torch::Tensor& x, const torch::Device& device) {
+  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.device() == device &&
+              x.is_contiguous());
+  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
+}
+
+}  // namespace
+
+void glue(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cos_sin,
+          torch::Tensor wq, torch::Tensor wkv, torch::Tensor qn,
+          torch::Tensor kvn, torch::Tensor kpe,
+          std::optional<torch::Tensor> cache1,
+          std::optional<torch::Tensor> slots1,
+          std::optional<torch::Tensor> cache2,
+          std::optional<torch::Tensor> slots2, double eps_q, double eps_kv) {
+  const auto device = qkv.device();
+  const int64_t m = qkv.size(0);
+  TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == LD && qkv.stride(1) == 1 &&
+              qkv.scalar_type() == torch::kBFloat16 && qkv.stride(0) % 8 == 0 &&
+              reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0);
+  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
+              positions.dim() == 1 && positions.size(0) >= m &&
+              positions.stride(0) == 1 && positions.device() == device);
+  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == PE);
+  check_bf16(cos_sin, device);
+  check_bf16(wq, device);
+  check_bf16(wkv, device);
+  TORCH_CHECK(wq.numel() == QL && wkv.numel() == KVL);
+  check_bf16(qn, device);
+  check_bf16(kvn, device);
+  check_bf16(kpe, device);
+  TORCH_CHECK(qn.numel() == m * QL && kvn.numel() == m * KVL &&
+              kpe.numel() == m * PE);
+  TORCH_CHECK(cache2.has_value() <= cache1.has_value());
+  if (m == 0) return;
+  const Cache c1 = cache_of(cache1, slots1, device);
+  const Cache c2 = cache_of(cache2, slots2, device);
+
+  const c10::cuda::CUDAGuard guard(device);
+  cudaLaunchConfig_t cfg = {};
+  cfg.gridDim = dim3(m);
+  cfg.blockDim = dim3(THREADS);
+  cfg.stream = at::cuda::getCurrentCUDAStream(device.index());
+  cudaLaunchAttribute attr;
+  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
+  attr.val.programmaticStreamSerializationAllowed = 1;
+  cfg.attrs = &attr;
+  cfg.numAttrs = 1;
+  auto B = [](const torch::Tensor& x) {
+    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
+  };
+  C10_CUDA_CHECK(cudaLaunchKernelEx(
+      &cfg, qkv_glue_kernel, B(qkv), qkv.stride(0),
+      positions.data_ptr<int64_t>(), B(cos_sin), cos_sin.size(0), B(wq),
+      B(wkv), B(qn), B(kvn), B(kpe), c1, c2, float(eps_q), float(eps_kv)));
+}
+
+PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("glue", &glue); }
diff --git a/tests/kernels/test_mla_qkv_glue.py b/tests/kernels/test_mla_qkv_glue.py
new file mode 100644
index 0000000000..f00787854a
--- /dev/null
+++ b/tests/kernels/test_mla_qkv_glue.py
@@ -0,0 +1,130 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""mla_glue.cu (vllm::mla_qkv_glue): the q_a / kv_a RMSNorms and k_pe RoPE
+must equal the served math (fp32, one bf16 rounding) up to the fp32 sum order,
+and its KV writes must be byte-identical to concat_and_cache_mla(fp8_ds_mla)
+of its own outputs, padded slots and untouched entries included."""
+
+import pytest
+import torch
+
+from vllm import _custom_ops as ops
+from vllm.platforms import current_platform
+
+pytestmark = pytest.mark.skipif(
+    not current_platform.is_device_capability(90), reason="sm_90a kernel"
+)
+
+QL, KVL, PE, ENTRY, EPS = 2048, 512, 64, 656, 1e-5
+MAX_POS, BLOCK = 4096, 64
+
+
+def reference(qkv, pos, cos_sin, wq, wkv):
+    q_c, kv_c, k_pe = qkv.float().split([QL, KVL, PE], -1)
+
+    def rms(x, w):
+        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
+        return (x * r * w.float()).to(torch.bfloat16)
+
+    cos, sin = cos_sin[pos].float().chunk(2, -1)
+    x0, x1 = k_pe[..., 0::2], k_pe[..., 1::2]
+    kpe = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), -1).flatten(-2)
+    return rms(q_c, wq), rms(kv_c, wkv), kpe.to(torch.bfloat16)
+
+
+def ulps(a, b):
+    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()
+
+
+@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 64, 512])
+def test_matches_served_math_and_cache_bytes(m):
+    from vllm.model_executor.layers.mla_glue import _extension
+
+    torch.manual_seed(m)
+    dev = "cuda"
+    # rows of varied scale, so tile scales and rounding boundaries vary
+    scale = torch.logspace(-3, 1, m, device=dev)[:, None]
+    qkv = (torch.randn(m, QL + KVL + PE, device=dev) * scale).to(torch.bfloat16)
+    pos = torch.randint(0, MAX_POS, (m,), device=dev)
+    angle = torch.rand(MAX_POS, PE // 2, device=dev) * 6.3
+    cos_sin = torch.cat((angle.cos(), angle.sin()), -1).to(torch.bfloat16)
+    wq = (1 + 0.1 * torch.randn(QL, device=dev)).to(torch.bfloat16)
+    wkv = (1 + 0.1 * torch.randn(KVL, device=dev)).to(torch.bfloat16)
+
+    blocks = (2 * m) // BLOCK + 2
+    cache = torch.full((blocks, BLOCK, ENTRY), 0xAB, dtype=torch.uint8, device=dev)
+    staged = torch.full((2, BLOCK, ENTRY), 0xCD, dtype=torch.uint8, device=dev)
+    # distinct slots, a padded token, and fewer slots than tokens (padding)
+    n = max(m - 1, 1)
+    slots = torch.randperm(blocks * BLOCK, device=dev)[:n]
+    slots[n // 2] = -1
+    rows = torch.arange(n, device=dev)
+    rows[rows >= 2 * BLOCK] = -1
+    rows[0] = -1  # a staged row the main write keeps
+
+    qn = torch.empty(m, QL, dtype=torch.bfloat16, device=dev)
+    kvn = torch.empty(m, KVL, dtype=torch.bfloat16, device=dev)
+    kpe = torch.empty(m, PE, dtype=torch.bfloat16, device=dev)
+    want_cache, want_staged = cache.clone(), staged.clone()
+    _extension().glue(
+        qkv, pos, cos_sin, wq, wkv, qn, kvn, kpe, cache, slots, staged, rows,
+        EPS, EPS,
+    )
+
+    rq, rkv, rpe = reference(qkv, pos, cos_sin, wq, wkv)
+    assert torch.equal(kpe, rpe)
+    for got, ref in ((qn, rq), (kvn, rkv)):
+        diff = ulps(got, ref)
+        assert diff.max() <= 1
+        assert (diff > 0).float().mean() < 1e-4
+
+    k_scale = torch.ones(1, device=dev)
+    ops.concat_and_cache_mla(kvn[:n], kpe[:n], want_cache, slots, "fp8_ds_mla", k_scale)
+    ops.concat_and_cache_mla(
+        kvn[:n], kpe[:n], want_staged, rows, "fp8_ds_mla", k_scale
+    )
+    assert torch.equal(cache, want_cache)
+    assert torch.equal(staged, want_staged)
+
+
+def test_without_caches_only_computes():
+    from vllm.model_executor.layers.mla_glue import _extension
+
+    dev = "cuda"
+    m = 8
+    qkv = torch.randn(m, QL + KVL + PE, device=dev).to(torch.bfloat16)
+    pos = torch.arange(m, device=dev)
+    cos_sin = torch.ones(MAX_POS, PE, dtype=torch.bfloat16, device=dev)
+    w = torch.ones(QL, dtype=torch.bfloat16, device=dev)
+    outs = [torch.empty(m, k, dtype=torch.bfloat16, device=dev) for k in (QL, KVL, PE)]
+    _extension().glue(
+        qkv, pos, cos_sin, w, w[:KVL], *outs, None, None, None, None, EPS, EPS
+    )
+    rq, rkv, _ = reference(qkv, pos, cos_sin, w, w[:KVL])
+    assert ulps(outs[0], rq).max() <= 1 and ulps(outs[1], rkv).max() <= 1
+
+
+def test_writes_kv_only_in_full_capture():
+    """The V2 runner captures FULL graphs from a NONE-mode forward: the glue
+    must write there (attention metadata, stream capturing) and nowhere else,
+    or the KV is written twice or not at all."""
+    from vllm.forward_context import ForwardContext, override_forward_context
+    from vllm.model_executor.layers.mla_glue import writes_kv
+
+    slots = {"l": torch.zeros(4, dtype=torch.long, device="cuda")}
+
+    def probe(attn_metadata, capture):
+        fc = ForwardContext(
+            no_compile_layers={}, attn_metadata=attn_metadata, slot_mapping=slots
+        )
+        with override_forward_context(fc):
+            if not capture:
+                return writes_kv()
+            graph, out = torch.cuda.CUDAGraph(), []
+            with torch.cuda.graph(graph):
+                out.append(writes_kv())
+            return out[0]
+
+    assert probe({"l": object()}, capture=True)
+    assert not probe(None, capture=True)  # piecewise capture
+    assert not probe({"l": object()}, capture=False)  # eager / piecewise replay
```

## Debug knobs used for the bisection (pf9 = pf8 + this)
```diff
diff -ru -x __pycache__ -x '*.so' pf8/vllm/model_executor/layers/mla_glue/__init__.py pf9/vllm/model_executor/layers/mla_glue/__init__.py
--- pf8/vllm/model_executor/layers/mla_glue/__init__.py	2026-10-18 16:15:06.122173000 +0200
+++ pf9/vllm/model_executor/layers/mla_glue/__init__.py	2026-10-18 16:56:45.973028000 +0200
@@ -49,6 +49,13 @@
     from vllm.platforms import current_platform
     from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
 
+    import os
+
+    from vllm.model_executor.models.utils import extract_layer_index
+
+    max_layer = int(os.environ.get("VLLM_MLA_GLUE_MAX_LAYER", "100000"))  # debug
+    if extract_layer_index(wrapper.prefix) >= max_layer:
+        return False
     rope = wrapper.rotary_emb
     norms = (wrapper.q_a_layernorm, wrapper.kv_a_layernorm)
     return (
diff -ru -x __pycache__ -x '*.so' pf8/vllm/model_executor/layers/mla_glue/mla_glue.cu pf9/vllm/model_executor/layers/mla_glue/mla_glue.cu
--- pf8/vllm/model_executor/layers/mla_glue/mla_glue.cu	2026-10-18 15:48:52.920887000 +0200
+++ pf9/vllm/model_executor/layers/mla_glue/mla_glue.cu	2026-10-18 16:56:45.971869000 +0200
@@ -212,7 +212,8 @@
   attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
   attr.val.programmaticStreamSerializationAllowed = 1;
   cfg.attrs = &attr;
-  cfg.numAttrs = 1;
+  static const bool pdl = getenv("VLLM_MLA_GLUE_NOPDL") == nullptr;  // debug
+  cfg.numAttrs = pdl ? 1 : 0;
   auto B = [](const torch::Tensor& x) {
     return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
   };
diff -ru -x __pycache__ -x '*.so' pf8/vllm/model_executor/layers/mla.py pf9/vllm/model_executor/layers/mla.py
--- pf8/vllm/model_executor/layers/mla.py	2026-10-18 15:48:52.919799000 +0200
+++ pf9/vllm/model_executor/layers/mla.py	2026-10-18 17:19:13.762965000 +0200
@@ -142,8 +142,9 @@
         if self.qkv_glue:
             assert self.fused_qkv_a_proj is not None
             assert self.q_a_layernorm is not None and self.q_b_proj is not None
+            qkv_dbg = self.fused_qkv_a_proj(hidden_states)[0]
             q_c, kv_c_normed, k_pe = torch.ops.vllm.mla_qkv_glue(
-                    self.fused_qkv_a_proj(hidden_states)[0],
+                    qkv_dbg,
                     positions,
                     self.rotary_emb.cos_sin_cache,
                     self.q_a_layernorm.weight,
@@ -152,11 +153,26 @@
                     self.kv_a_layernorm.variance_epsilon,
                     _encode_layer_name(self.mla_attn.layer_name),
             )
+            import os
+
+            if os.environ.get("VLLM_MLA_GLUE_QORIG") == "1":  # debug
+                q_c = self.q_a_layernorm(qkv_dbg[..., : self.q_lora_rank])
+            if os.environ.get("VLLM_MLA_GLUE_KVORIG") == "1":  # debug
+                kv_c_normed = self.kv_a_layernorm(
+                    qkv_dbg[..., self.q_lora_rank : self.q_lora_rank + self.kv_lora_rank]
+                )
             q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
             k_pe = k_pe.unsqueeze(1)
-            q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
-                positions, q[..., self.qk_nope_head_dim :]
-            )
+
+            if os.environ.get("VLLM_MLA_GLUE_KROPE", "1") == "0":  # debug
+                k_pe = qkv_dbg[..., -self.qk_rope_head_dim :].unsqueeze(1)
+                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
+                    positions, q[..., self.qk_nope_head_dim :], k_pe
+                )
+            else:
+                q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
+                    positions, q[..., self.qk_nope_head_dim :]
+                )
         elif self.q_lora_rank is not None:
             assert self.fused_qkv_a_proj is not None, (
                 "fused_qkv_a_proj is required when q_lora_rank is not None"
```

## pf9 vllm/model_executor/layers/mla.py lines 1-237 (the MLA wrapper; q_c also feeds the DSA indexer at line ~223)
```python
     1	# SPDX-License-Identifier: Apache-2.0
     2	# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
     3	from dataclasses import dataclass
     4	
     5	import torch
     6	
     7	from vllm.config import CacheConfig
     8	from vllm.model_executor.custom_op import PluggableLayer
     9	from vllm.model_executor.layers.attention import MLAAttention
    10	from vllm.model_executor.layers.quantization import QuantizationConfig
    11	from vllm.utils.torch_utils import _encode_layer_name
    12	
    13	
    14	@dataclass
    15	class MLAModules:
    16	    """Modules used in MLA."""
    17	
    18	    kv_a_layernorm: torch.nn.Module
    19	    kv_b_proj: torch.nn.Module
    20	    rotary_emb: torch.nn.Module
    21	    o_proj: torch.nn.Module
    22	    fused_qkv_a_proj: torch.nn.Module | None
    23	    kv_a_proj_with_mqa: torch.nn.Module | None
    24	    q_a_layernorm: torch.nn.Module | None
    25	    q_b_proj: torch.nn.Module | None
    26	    q_proj: torch.nn.Module | None
    27	    indexer: torch.nn.Module | None
    28	    is_sparse: bool
    29	    topk_indices_buffer: torch.Tensor | None
    30	    indexer_rotary_emb: torch.nn.Module | None = None
    31	
    32	
    33	# --8<-- [start:multi_head_latent_attention]
    34	@PluggableLayer.register("multi_head_latent_attention")
    35	class MultiHeadLatentAttentionWrapper(PluggableLayer):
    36	    """Pluggable MLA layer which allows OOT backends to add
    37	    custom implementations of the outer MLA layer (including rope & o_proj).
    38	    Note that currently oot platforms can still use CustomOp.register_oot to
    39	    replace MLA layer entirely, although we use PluggableLayer to register
    40	    this layer now.
    41	
    42	    This class takes positions and hidden_states as input.
    43	    The input tensors can either contain prefill tokens or decode tokens.
    44	    The class does the following:
    45	
    46	    1. MLA Preprocess.
    47	    2. Perform multi-head attention to prefill tokens and
    48	       multi-query attention to decode tokens separately.
    49	    3. Return the output tensor.
    50	    """
    51	
    52	    # --8<-- [end:multi_head_latent_attention]
    53	
    54	    def __init__(
    55	        self,
    56	        hidden_size: int,
    57	        num_heads: int,
    58	        scale: float,
    59	        qk_nope_head_dim: int,
    60	        qk_rope_head_dim: int,
    61	        v_head_dim: int,
    62	        q_lora_rank: int | None,
    63	        kv_lora_rank: int,
    64	        mla_modules: MLAModules,
    65	        cache_config: CacheConfig | None = None,
    66	        quant_config: QuantizationConfig | None = None,
    67	        prefix: str = "",
    68	        skip_topk: bool = False,
    69	    ) -> None:
    70	        super().__init__()
    71	        self.hidden_size = hidden_size
    72	        self.qk_nope_head_dim = qk_nope_head_dim
    73	        self.qk_rope_head_dim = qk_rope_head_dim
    74	        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    75	        self.v_head_dim = v_head_dim
    76	        self.q_lora_rank = q_lora_rank
    77	        self.kv_lora_rank = kv_lora_rank
    78	        self.num_heads = num_heads
    79	        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
    80	        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
    81	        self.q_a_layernorm = mla_modules.q_a_layernorm
    82	        self.q_b_proj = mla_modules.q_b_proj
    83	        self.q_proj = mla_modules.q_proj
    84	        self.kv_a_layernorm = mla_modules.kv_a_layernorm
    85	        self.kv_b_proj = mla_modules.kv_b_proj
    86	        self.rotary_emb = mla_modules.rotary_emb
    87	        self.o_proj = mla_modules.o_proj
    88	        self.indexer = mla_modules.indexer
    89	        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
    90	        self.is_sparse = mla_modules.is_sparse
    91	
    92	        # Whether to skip top-k token selection computation in this layer.
    93	        # When True, the indexer will not be called, and the layer will reuse
    94	        # the topk_tokens buffer written by a previous layer in the same pass.
    95	        # Refer: https://arxiv.org/abs/2603.12201 for more details.
    96	        self.skip_topk = skip_topk
    97	        if self.indexer is not None:
    98	            assert hasattr(self.indexer, "topk_tokens")
    99	            self.topk_tokens = self.indexer.topk_tokens
   100	            self.topk_indices_buffer = mla_modules.topk_indices_buffer
   101	
   102	        self.mla_attn = MLAAttention(
   103	            num_heads=self.num_heads,
   104	            scale=scale,
   105	            qk_nope_head_dim=self.qk_nope_head_dim,
   106	            qk_rope_head_dim=self.qk_rope_head_dim,
   107	            v_head_dim=self.v_head_dim,
   108	            q_lora_rank=self.q_lora_rank,
   109	            kv_lora_rank=self.kv_lora_rank,
   110	            cache_config=cache_config,
   111	            quant_config=quant_config,
   112	            prefix=f"{prefix}.attn",
   113	            kv_b_proj=self.kv_b_proj,
   114	            use_sparse=self.is_sparse,
   115	            indexer=self.indexer,
   116	            topk_indices_buffer=mla_modules.topk_indices_buffer,
   117	        )
   118	        # read by the attention op to prefetch o_proj (VLLM_DENSE_L2_PREFETCH)
   119	        self.mla_attn.l2_prefetch_o_proj = (self.o_proj,)
   120	
   121	        self.prefix = prefix
   122	        from vllm.model_executor.model_loader.tiered_moe_prefetch import (
   123	            cold_prefetch_layer,
   124	        )
   125	
   126	        self.cold_prefetch_layer = cold_prefetch_layer(prefix)
   127	
   128	        from vllm.model_executor.layers import mla_glue
   129	
   130	        self.qkv_glue = mla_glue.supported(self)
   131	        self.mla_attn.qkv_glue = self.qkv_glue
   132	
   133	    def forward(
   134	        self,
   135	        positions: torch.Tensor,
   136	        hidden_states: torch.Tensor,
   137	        llama_4_scaling: torch.Tensor | None = None,
   138	    ) -> torch.Tensor:
   139	        q_c = None
   140	        kv_lora = None
   141	
   142	        if self.qkv_glue:
   143	            assert self.fused_qkv_a_proj is not None
   144	            assert self.q_a_layernorm is not None and self.q_b_proj is not None
   145	            qkv_dbg = self.fused_qkv_a_proj(hidden_states)[0]
   146	            q_c, kv_c_normed, k_pe = torch.ops.vllm.mla_qkv_glue(
   147	                    qkv_dbg,
   148	                    positions,
   149	                    self.rotary_emb.cos_sin_cache,
   150	                    self.q_a_layernorm.weight,
   151	                    self.kv_a_layernorm.weight,
   152	                    self.q_a_layernorm.variance_epsilon,
   153	                    self.kv_a_layernorm.variance_epsilon,
   154	                    _encode_layer_name(self.mla_attn.layer_name),
   155	            )
   156	            import os
   157	
   158	            if os.environ.get("VLLM_MLA_GLUE_QORIG") == "1":  # debug
   159	                q_c = self.q_a_layernorm(qkv_dbg[..., : self.q_lora_rank])
   160	            if os.environ.get("VLLM_MLA_GLUE_KVORIG") == "1":  # debug
   161	                kv_c_normed = self.kv_a_layernorm(
   162	                    qkv_dbg[..., self.q_lora_rank : self.q_lora_rank + self.kv_lora_rank]
   163	                )
   164	            q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
   165	            k_pe = k_pe.unsqueeze(1)
   166	
   167	            if os.environ.get("VLLM_MLA_GLUE_KROPE", "1") == "0":  # debug
   168	                k_pe = qkv_dbg[..., -self.qk_rope_head_dim :].unsqueeze(1)
   169	                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
   170	                    positions, q[..., self.qk_nope_head_dim :], k_pe
   171	                )
   172	            else:
   173	                q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
   174	                    positions, q[..., self.qk_nope_head_dim :]
   175	                )
   176	        elif self.q_lora_rank is not None:
   177	            assert self.fused_qkv_a_proj is not None, (
   178	                "fused_qkv_a_proj is required when q_lora_rank is not None"
   179	            )
   180	            assert self.q_a_layernorm is not None, (
   181	                "q_a_layernorm is required when q_lora_rank is not None"
   182	            )
   183	            assert self.q_b_proj is not None, (
   184	                "q_b_proj is required when q_lora_rank is not None"
   185	            )
   186	
   187	            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
   188	            q_c, kv_lora = qkv_lora.split(
   189	                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
   190	                dim=-1,
   191	            )
   192	            q_c = self.q_a_layernorm(q_c)
   193	            q = self.q_b_proj(q_c)[0]
   194	        else:
   195	            assert self.kv_a_proj_with_mqa is not None, (
   196	                "kv_a_proj_with_mqa is required when q_lora_rank is None"
   197	            )
   198	            assert self.q_proj is not None, (
   199	                "q_proj is required when q_lora_rank is None"
   200	            )
   201	            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
   202	            q = self.q_proj(hidden_states)[0]
   203	
   204	        if not self.qkv_glue:
   205	            kv_c, k_pe = kv_lora.split(
   206	                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
   207	            )
   208	            kv_c_normed = self.kv_a_layernorm(kv_c)
   209	
   210	            q = q.view(-1, self.num_heads, self.qk_head_dim)
   211	            # Add head dim of 1 to k_pe
   212	            k_pe = k_pe.unsqueeze(1)
   213	
   214	            if self.rotary_emb is not None:
   215	                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
   216	                    positions, q[..., self.qk_nope_head_dim :], k_pe
   217	                )
   218	
   219	        if self.cold_prefetch_layer is not None:
   220	            torch.ops.vllm.tiered_moe_cold_join(q)
   221	
   222	        if self.indexer and self.is_sparse and not self.skip_topk:
   223	            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
   224	
   225	        if llama_4_scaling is not None:
   226	            q *= llama_4_scaling
   227	
   228	        attn_out = self.mla_attn(
   229	            q,
   230	            kv_c_normed,
   231	            k_pe,
   232	            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
   233	        )
   234	
   235	        if self.cold_prefetch_layer is not None:
   236	            torch.ops.vllm.tiered_moe_cold_fork(attn_out, self.cold_prefetch_layer)
   237	        return self.o_proj(attn_out)[0]
```

## pf9 vllm/model_executor/models/deepseek_v2.py Indexer (lines 647-830)
```python
   647	class Indexer(nn.Module):
   648	    def __init__(
   649	        self,
   650	        vllm_config: VllmConfig,
   651	        config: DeepseekV2Config | DeepseekV3Config,
   652	        hidden_size: int,
   653	        q_lora_rank: int,
   654	        quant_config: QuantizationConfig | None,
   655	        cache_config: CacheConfig | None,
   656	        topk_indices_buffer: torch.Tensor | None,
   657	        prefix: str = "",
   658	        is_inplace_rope: bool = False,
   659	    ):
   660	        super().__init__()
   661	        self.vllm_config = vllm_config
   662	        self.config = config
   663	        self.quant_config = quant_config
   664	        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
   665	        self.topk_tokens = config.index_topk
   666	        self.n_head = config.index_n_heads  # 64
   667	        self.head_dim = config.index_head_dim  # 128
   668	        self.rope_dim = config.qk_rope_head_dim  # 64
   669	        self.q_lora_rank = q_lora_rank  # 1536
   670	        # no tensor parallel, just replicated
   671	        self.wq_b = ReplicatedLinear(
   672	            self.q_lora_rank,
   673	            self.head_dim * self.n_head,
   674	            bias=False,
   675	            quant_config=quant_config,
   676	            prefix=f"{prefix}.wq_b",
   677	        )
   678	        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
   679	        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
   680	        self.wk_weights_proj = MergedColumnParallelLinear(
   681	            hidden_size,
   682	            [self.head_dim, self.n_head],
   683	            bias=False,
   684	            quant_config=None,
   685	            disable_tp=True,
   686	            prefix=f"{prefix}.wk_weights_proj",
   687	        )
   688	        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
   689	        self.softmax_scale = self.head_dim**-0.5
   690	
   691	        self.scale_fmt = "ue8m0"
   692	        self.quant_block_size = 128  # TODO: get from config
   693	        self.topk_indices_buffer = topk_indices_buffer
   694	
   695	        # NOTE: (zyongye) we use fp8 naive cache,
   696	        #       where we store value in fp8 and scale in fp32
   697	        #       per self.quant_block_size element
   698	        self.k_cache = DeepseekV32IndexerCache(
   699	            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
   700	            dtype=torch.uint8,
   701	            prefix=f"{prefix}.k_cache",
   702	            cache_config=cache_config,
   703	        )
   704	        self.max_model_len = vllm_config.model_config.max_model_len
   705	        self.prefix = prefix
   706	        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
   707	
   708	        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
   709	        self.indexer_op = SparseAttnIndexer(
   710	            self.k_cache,
   711	            self.quant_block_size,
   712	            self.scale_fmt,
   713	            self.topk_tokens,
   714	            self.head_dim,
   715	            self.max_model_len,
   716	            self.max_total_seq_len,
   717	            self.topk_indices_buffer,
   718	        )
   719	
   720	        self.is_inplace_rope = is_inplace_rope
   721	        self.n_head_scale = self.n_head**-0.5
   722	        self.use_fused_indexer_q = (
   723	            current_platform.is_cuda()
   724	            and self.quant_block_size == self.head_dim
   725	            and self.head_dim == 128
   726	            and self.rope_dim == 64
   727	            and self.scale_fmt is not None
   728	        )
   729	        self.index_trace_capture_fn: Callable[[torch.Tensor], None] | None = None
   730	
   731	    def set_index_trace_capture_fn(
   732	        self, capture_fn: Callable[[torch.Tensor], None]
   733	    ) -> None:
   734	        self.index_trace_capture_fn = capture_fn
   735	
   736	    def _capture_index_trace(self, topk_indices: torch.Tensor) -> torch.Tensor:
   737	        if self.index_trace_capture_fn is not None:
   738	            self.index_trace_capture_fn(topk_indices)
   739	        return topk_indices
   740	
   741	    def forward(
   742	        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
   743	    ) -> torch.Tensor:
   744	        q, _ = self.wq_b(qr)
   745	        q = q.view(-1, self.n_head, self.head_dim)
   746	
   747	        if current_platform.is_rocm() and self.is_inplace_rope:
   748	            # This path should works on all platform, will remove extra
   749	            # branches in the future
   750	            # This fast path relies on rotary_emb mutating q and k inplace.
   751	            # On ROCm, this is only valid for kernels used as custom ops.
   752	            # In pytorch-native rope for inductor fusion, rotated q/k tensors
   753	            # are not mutated inplace but returned as new tensors.
   754	            # Fused wk + weights_proj: one GEMM, then split
   755	            kw, _ = self.wk_weights_proj(hidden_states)
   756	            k = kw[:, : self.head_dim]
   757	            weights = kw[:, self.head_dim :]
   758	
   759	            k = self.k_norm(k)
   760	
   761	            rotary_emb(
   762	                positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
   763	            )
   764	        elif self.use_fused_indexer_q and q.dtype == torch.bfloat16:
   765	            # fused wk + weights_proj: one GEMM, then split
   766	            kw, _ = self.wk_weights_proj(hidden_states)
   767	            k = kw[:, : self.head_dim]
   768	            weights = kw[:, self.head_dim :]
   769	
   770	            k = self.k_norm(k)
   771	            k_pe, k_nope = torch.split(
   772	                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
   773	            )
   774	
   775	            q_fp8, weights = fused_indexer_q_rope_quant(
   776	                positions,
   777	                q,
   778	                rotary_emb.cos_sin_cache,
   779	                weights,
   780	                self.softmax_scale,
   781	                self.n_head_scale,
   782	                rotary_emb.is_neox_style,
   783	            )
   784	
   785	            # rotate only the MQA K
   786	            k_pe = k_pe.unsqueeze(1)
   787	            q_dummy = torch.empty_like(k_pe)
   788	            _, k_pe = rotary_emb(positions, q_dummy, k_pe)
   789	            k_pe = k_pe.reshape(-1, self.rope_dim)
   790	            k = torch.cat([k_pe, k_nope], dim=-1)
   791	
   792	            return self._capture_index_trace(
   793	                self.indexer_op(hidden_states, q_fp8, k, weights)
   794	            )
   795	        else:
   796	            q_pe, q_nope = torch.split(
   797	                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
   798	            )
   799	            # Fused wk + weights_proj: one GEMM, then split
   800	            kw, _ = self.wk_weights_proj(hidden_states)
   801	            k = kw[:, : self.head_dim]
   802	            weights = kw[:, self.head_dim :]
   803	
   804	            k = self.k_norm(k)
   805	            k_pe, k_nope = torch.split(
   806	                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
   807	            )
   808	
   809	            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
   810	            # Note: RoPE (NeoX) can introduce extra leading dimensions during
   811	            # compilation so we need to reshape back to token-flattened shapes
   812	            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
   813	            k_pe = k_pe.reshape(-1, self.rope_dim)
   814	
   815	            # `rotary_emb` is shape-preserving; `q_pe` is already
   816	            # [num_tokens, n_head, rope_dim].
   817	            q = torch.cat([q_pe, q_nope], dim=-1)
   818	            # `k_pe` is [num_tokens, rope_dim] (MQA).
   819	            k = torch.cat([k_pe, k_nope], dim=-1)
   820	
   821	        # we only quant q here since k quant is fused with cache insertion
   822	        q = q.view(-1, self.head_dim)
   823	        q_fp8, q_scale = per_token_group_quant_fp8(
   824	            q,
   825	            self.quant_block_size,
   826	            column_major_scales=False,
   827	            use_ue8m0=self.scale_fmt is not None,
   828	        )
   829	        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
   830	        q_scale = q_scale.view(-1, self.n_head)
```

## pf9 vllm/model_executor/layers/attention/mla_attention.py lines 575-700, 860-900, 1100-1160
```python
   575	        if self._chunked_prefill_workspace_size is None:
   576	            self._chunked_prefill_workspace_size = (
   577	                MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
   578	                    self._vllm_config
   579	                )
   580	            )
   581	        return self._chunked_prefill_workspace_size
   582	
   583	    def forward(
   584	        self,
   585	        q: torch.Tensor,
   586	        kv_c_normed: torch.Tensor,
   587	        k_pe: torch.Tensor,
   588	        output_shape: torch.Size | None = None,
   589	    ) -> torch.Tensor:
   590	        if self.calculate_kv_scales:
   591	            torch.ops.vllm.maybe_calc_kv_scales(
   592	                q,
   593	                kv_c_normed,
   594	                k_pe,
   595	                _encode_layer_name(self.layer_name),
   596	            )
   597	
   598	        if self.use_direct_call:
   599	            forward_context: ForwardContext = get_forward_context()
   600	            attn_metadata_raw = forward_context.attn_metadata
   601	            attn_metadata: MLACommonMetadata
   602	            if isinstance(attn_metadata_raw, dict):
   603	                attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
   604	            elif isinstance(attn_metadata_raw, list):
   605	                # list[dict[str, AttentionMetadata]]: used in speculative decoding
   606	                # where [0] is the base-model (non-speculative) metadata dict.
   607	                attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
   608	            else:
   609	                attn_metadata = attn_metadata_raw
   610	            self_kv_cache = self.kv_cache
   611	            slot_mapping = forward_context.slot_mapping
   612	
   613	            assert isinstance(slot_mapping, dict), (
   614	                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
   615	            )
   616	            if not _glue_wrote_kv(self):
   617	                self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
   618	                    kv_c_normed,
   619	                    k_pe,
   620	                    self_kv_cache,
   621	                    slot_mapping.get(self.layer_name),
   622	                    self.kv_cache_dtype,
   623	                    self._k_scale,
   624	                )
   625	            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
   626	            self.forward_impl(
   627	                q,
   628	                kv_c_normed,
   629	                k_pe,
   630	                self_kv_cache,
   631	                attn_metadata,
   632	                output=output,
   633	            )
   634	            return output
   635	        else:
   636	            encoded = _encode_layer_name(self.layer_name)
   637	            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
   638	                kv_c_normed,
   639	                k_pe,
   640	                encoded,
   641	                self.kv_cache_dtype,
   642	                self._k_scale,
   643	            )
   644	            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
   645	            torch.ops.vllm.unified_mla_attention_with_output(
   646	                q,
   647	                kv_c_normed,
   648	                k_pe,
   649	                output,
   650	                encoded,
   651	                kv_cache_dummy_dep=kv_cache_dummy_dep,
   652	            )
   653	            return output
   654	
   655	    def forward_impl(
   656	        self,
   657	        q: torch.Tensor,
   658	        k_c_normed: torch.Tensor,  # key in unified attn
   659	        k_pe: torch.Tensor,  # value in unified attn
   660	        kv_cache: torch.Tensor,
   661	        attn_metadata: "MLACommonMetadata",
   662	        output: torch.Tensor,
   663	        output_scale: torch.Tensor | None = None,
   664	        output_block_scale: torch.Tensor | None = None,
   665	        quant_group_size: int | None = None,
   666	        quant_scale_ue8m0: bool | None = None,
   667	        quant_col_major: bool | None = None,
   668	        quant_tma_aligned: bool | None = None,
   669	    ) -> torch.Tensor:
   670	        assert output is not None, "Output tensor must be provided."
   671	
   672	        quant_key = _detect_output_quant_key(
   673	            output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
   674	        )
   675	        if quant_key is not None:
   676	            # The fusion pass has allocated output with quantized dtype
   677	            # (FP8 or uint8 for FP4). We can't write into it directly,
   678	            # so we swap in a temp buffer for computation, then quantize
   679	            # into the real output at the end.
   680	            # NOTE(carlyou): this is temporary until kernels support fp8 output
   681	            quant_output = output
   682	            output = torch.empty(
   683	                output.shape[0],
   684	                self.num_heads * self.v_head_dim,
   685	                dtype=q.dtype,
   686	                device=output.device,
   687	            )
   688	
   689	        if attn_metadata is None:
   690	            # During the profile run try to simulate to worse case output size
   691	            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
   692	            # since this can be large
   693	            _ = torch.empty(
   694	                (
   695	                    self.chunked_prefill_workspace_size,
   696	                    self.num_heads,
   697	                    self.qk_nope_head_dim + self.v_head_dim,
   698	                ),
   699	                device=k_c_normed.device,
   700	                dtype=k_c_normed.dtype,
   860	                mqa_q = (mqa_ql_nope, mqa_q_pe)
   861	            if q_head_major is not None:
   862	                q_head_major[..., self.kv_lora_rank :].copy_(mqa_q_pe.transpose(0, 1))
   863	                mqa_q = get_dcp_group().all_gather(q_head_major, dim=0).transpose(0, 1)
   864	            elif self.impl.dcp_world_size > 1:
   865	                # all-gather in the head dim; a (nope, pe) pair -> (B, N, L + P)
   866	                mqa_q = cp_all_gather_q(mqa_q, get_dcp_group())
   867	
   868	            # call decode attn
   869	            if not self.impl.is_sparse:
   870	                assert attn_metadata.decode is not None
   871	            attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]
   872	            o_proj = getattr(self, "l2_prefetch_o_proj", None)
   873	            if envs.VLLM_DENSE_L2_PREFETCH and o_proj is not None:
   874	                from vllm.model_executor.layers.l2_prefetch import (
   875	                    prefetch_on_side_stream,
   876	                )
   877	
   878	                # bandwidth is idle until o_proj (DCP combine, W_UV); W_UV
   879	                # first, or it waits behind the o_proj prefetch
   880	                if envs.VLLM_DENSE_L2_PREFETCH == 2:
   881	                    prefetch_on_side_stream(self.W_UV, grid=32, threads=32)
   882	                prefetch_on_side_stream(o_proj[0].weight, grid=32, threads=32)
   883	
   884	            # correct dcp attn_out with lse.
   885	            if self.impl.dcp_world_size > 1:
   886	                if self.dcp_a2a:
   887	                    attn_out = dcp_a2a_lse_reduce(
   888	                        attn_out,
   889	                        lse,
   890	                        get_dcp_group(),
   891	                        is_lse_base_on_e=self.impl.lse_base_on_e,
   892	                    )
   893	                else:
   894	                    # head-major: v_up_proj transposes to heads first anyway
   895	                    attn_out = cp_lse_ag_out_rs(
   896	                        attn_out,
   897	                        lse,
   898	                        get_dcp_group(),
   899	                        is_lse_base_on_e=self.impl.lse_base_on_e,
   900	                        head_major_out=True,
  1100	        elif self.is_aiter_triton_fp8_bmm_enabled:
  1101	            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
  1102	            x = rocm_aiter_ops.triton_fp8_bmm(
  1103	                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
  1104	            )
  1105	        else:
  1106	            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
  1107	            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
  1108	
  1109	
  1110	def _glue_wrote_kv(layer: "MLAAttention") -> bool:
  1111	    """vllm::mla_qkv_glue wrote the layer's KV itself (FULL graph capture)."""
  1112	    from vllm.model_executor.layers import mla_glue
  1113	
  1114	    return getattr(layer, "qkv_glue", False) and mla_glue.writes_kv()
  1115	
  1116	
  1117	def unified_mla_kv_cache_update(
  1118	    kv_c_normed: torch.Tensor,
  1119	    k_pe: torch.Tensor,
  1120	    layer_name: LayerNameType,
  1121	    kv_cache_dtype: str,
  1122	    k_scale: torch.Tensor,
  1123	) -> torch.Tensor:
  1124	    """
  1125	    Returns a dummy that is passed to unified_attention to signal a side effect and
  1126	    the data dependency between them to ensure torch.compile preserves ordering.
  1127	    """
  1128	    layer_name = _resolve_layer_name(layer_name)
  1129	    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
  1130	    if layer_slot_mapping is not None and not _glue_wrote_kv(attn_layer):
  1131	        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
  1132	            kv_c_normed,
  1133	            k_pe,
  1134	            kv_cache,
  1135	            layer_slot_mapping,
  1136	            kv_cache_dtype,
  1137	            k_scale,
  1138	        )
  1139	
  1140	    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)
  1141	
  1142	
  1143	def unified_mla_kv_cache_update_fake(
  1144	    kv_c_normed: torch.Tensor,
  1145	    k_pe: torch.Tensor,
  1146	    layer_name: LayerNameType,
  1147	    kv_cache_dtype: str,
  1148	    k_scale: torch.Tensor,
  1149	) -> torch.Tensor:
  1150	    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)
  1151	
  1152	
  1153	direct_register_custom_op(
  1154	    op_name="unified_mla_kv_cache_update",
  1155	    op_func=unified_mla_kv_cache_update,
  1156	    fake_impl=unified_mla_kv_cache_update_fake,
  1157	)
  1158	
  1159	
  1160	@eager_break_during_capture
```
