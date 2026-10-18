You are reviewing a vLLM integration of a fused CUDA kernel ("glue2") for GLM-5.3 MLA decode on GH200 (sm_90a). Do NOT run any commands; everything you need is inlined. Review for correctness bugs first (ordering, CUDA graph capture/replay, torch.compile splitting, KV-cache/skip-KV staging semantics, stream ordering, eligibility), then numerics vs the served path, then performance. Be concrete: file:line, failure scenario, fix. Rank findings by severity, and end with go/no-go for a served A/B and for default enablement.

Context:
- Served config: TP4 + DCP4, MTP3 speculative decode, cudagraph_mode FULL_AND_PIECEWISE (decode = FULL graphs captured per size up to 4c tokens; prefill/mixed = piecewise graphs), splitting_ops default (attention ops + unified_kv_cache_update + unified_mla_kv_cache_update), torch 2.11, model runner V2. KV cache dtype fp8_ds_mla (656-byte entries). Index-share ("skip") layers keep their main MLA cache on Grace (pinned, UVA) and are staged into HBM per decode step by skip_kv_stage.py (inlined below); their own new tokens go to reserved rows of the staged buffer via write_new.
- Before: per layer, inductor emits triton kernels for k_pe RoPE and the q_a/kv_a RMSNorms between fused_qkv_a and q_b; then unified_mla_kv_cache_update (a splitting op) calls do_kv_cache_update = concat_and_cache_mla into the main cache, plus write_new for skip layers.
- After (VLLM_MLA_QKV_GLUE=1): one op vllm::mla_qkv_glue (registered as a splitting op) does both norms, k_pe RoPE and both cache writes, and returns a dummy dep tensor that replaces unified_mla_kv_cache_update's; q RoPE is still applied by rotary_emb on q only (key=None).
- Prior Astra review of the standalone kernel (summary): layout/alignment/rounding OK; asked for stream-ordering audit of moving skip writes earlier, mutation visibility to compile, per-invocation suppression, eligibility guards, negative positions, full PDL chain checks, served A/B.
- Unit test results (tests/kernels/test_mla_qkv_glue.py, inlined): pass for M in 1,4,8,16,32,64,512: kpe bitwise vs fp32-math reference, qn/kvn <= 1 bf16 ulp and <1e-4 of elements differ, cache bytes identical to ops.concat_and_cache_mla of the kernel's own outputs (both caches, padded slots, fewer slots than tokens, untouched sentinel bytes).

## The diff (git diff against the base, plus new files)
```diff
diff --git a/vllm/config/compilation.py b/vllm/config/compilation.py
index 810c40131f..be632a0222 100644
--- a/vllm/config/compilation.py
+++ b/vllm/config/compilation.py
@@ -1139,6 +1139,9 @@ class CompilationConfig:
                         self.pass_config.fuse_rope_kvcache = False
                     self.splitting_ops.append("vllm::unified_kv_cache_update")
                     self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
+                    if envs.VLLM_MLA_QKV_GLUE:
+                        # writes the MLA KV cache in its place
+                        self.splitting_ops.append("vllm::mla_qkv_glue")
 
             elif len(self.splitting_ops) == 0:
                 if (
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
index c394cc7a88..19bf81dd77 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -586,7 +586,10 @@ class MLAAttention(nn.Module, AttentionLayerBase):
         kv_c_normed: torch.Tensor,
         k_pe: torch.Tensor,
         output_shape: torch.Size | None = None,
+        kv_cache_dummy_dep: torch.Tensor | None = None,
     ) -> torch.Tensor:
+        """kv_cache_dummy_dep: from an op that already wrote this step's KV
+        (vllm::mla_qkv_glue); the cache update is then skipped."""
         if self.calculate_kv_scales:
             torch.ops.vllm.maybe_calc_kv_scales(
                 q,
@@ -613,14 +616,15 @@ class MLAAttention(nn.Module, AttentionLayerBase):
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
+            if kv_cache_dummy_dep is None:
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
@@ -633,13 +637,14 @@ class MLAAttention(nn.Module, AttentionLayerBase):
             return output
         else:
             encoded = _encode_layer_name(self.layer_name)
-            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
-                kv_c_normed,
-                k_pe,
-                encoded,
-                self.kv_cache_dtype,
-                self._k_scale,
-            )
+            if kv_cache_dummy_dep is None:
+                kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
+                    kv_c_normed,
+                    k_pe,
+                    encoded,
+                    self.kv_cache_dtype,
+                    self._k_scale,
+                )
             output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
             torch.ops.vllm.unified_mla_attention_with_output(
                 q,
diff --git a/vllm/model_executor/layers/mla.py b/vllm/model_executor/layers/mla.py
index 2bcf9127e0..e749de9d60 100644
--- a/vllm/model_executor/layers/mla.py
+++ b/vllm/model_executor/layers/mla.py
@@ -8,6 +8,7 @@ from vllm.config import CacheConfig
 from vllm.model_executor.custom_op import PluggableLayer
 from vllm.model_executor.layers.attention import MLAAttention
 from vllm.model_executor.layers.quantization import QuantizationConfig
+from vllm.utils.torch_utils import _encode_layer_name
 
 
 @dataclass
@@ -124,6 +125,10 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
 
         self.cold_prefetch_layer = cold_prefetch_layer(prefix)
 
+        from vllm.model_executor.layers import mla_glue
+
+        self.qkv_glue = mla_glue.supported(self)
+
     def forward(
         self,
         positions: torch.Tensor,
@@ -132,8 +137,29 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
     ) -> torch.Tensor:
         q_c = None
         kv_lora = None
-
-        if self.q_lora_rank is not None:
+        kv_cache_dummy_dep = None
+
+        if self.qkv_glue:
+            assert self.fused_qkv_a_proj is not None
+            assert self.q_a_layernorm is not None and self.q_b_proj is not None
+            q_c, kv_c_normed, k_pe, kv_cache_dummy_dep = (
+                torch.ops.vllm.mla_qkv_glue(
+                    self.fused_qkv_a_proj(hidden_states)[0],
+                    positions,
+                    self.rotary_emb.cos_sin_cache,
+                    self.q_a_layernorm.weight,
+                    self.kv_a_layernorm.weight,
+                    self.q_a_layernorm.variance_epsilon,
+                    self.kv_a_layernorm.variance_epsilon,
+                    _encode_layer_name(self.mla_attn.layer_name),
+                )
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
@@ -161,17 +187,20 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
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
@@ -187,6 +216,7 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
             kv_c_normed,
             k_pe,
             output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
+            kv_cache_dummy_dep=kv_cache_dummy_dep,
         )
 
         if self.cold_prefetch_layer is not None:
diff --git a/vllm/model_executor/layers/mla_glue/__init__.py b/vllm/model_executor/layers/mla_glue/__init__.py
new file mode 100644
index 0000000000..9176f5668d
--- /dev/null
+++ b/vllm/model_executor/layers/mla_glue/__init__.py
@@ -0,0 +1,159 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""MLA's work between fused_qkv_a and q_b in one kernel (mla_glue.cu): the
+q_a / kv_a RMSNorms, the k_pe RoPE and the layer's fp8_ds_mla KV writes
+(main cache, and the skip-KV stage's reserved rows on index-share layers).
+Registered as vllm::mla_qkv_glue, a splitting op in place of
+unified_mla_kv_cache_update: it reads the cache and slot mapping from the
+forward context the same way. VLLM_MLA_QKV_GLUE=1; built on first use for
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
+
+    rope = wrapper.rotary_emb
+    norms = (wrapper.q_a_layernorm, wrapper.kv_a_layernorm)
+    return (
+        envs.VLLM_MLA_QKV_GLUE
+        and current_platform.is_device_capability(90)
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
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
+    m = qkv.shape[0]
+    qn = qkv.new_empty(m, Q_LORA)
+    kvn = qkv.new_empty(m, KV_LORA)
+    kpe = qkv.new_empty(m, ROPE)
+    caches = (
+        _caches(_resolve_layer_name(layer_name), m)
+        if get_forward_context().slot_mapping
+        else (None,) * 4
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
+    return qn, kvn, kpe, qkv.new_empty(0)
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
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
+    m = qkv.shape[0]
+    return (
+        qkv.new_empty(m, Q_LORA),
+        qkv.new_empty(m, KV_LORA),
+        qkv.new_empty(m, ROPE),
+        qkv.new_empty(0),
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
index 0000000000..fe7c1f8268
--- /dev/null
+++ b/tests/kernels/test_mla_qkv_glue.py
@@ -0,0 +1,104 @@
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
```
## vllm/v1/attention/backends/mla/skip_kv_stage.py (unchanged, for context)
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM staging for index-share layers whose main MLA cache lives on Grace.

With ``--mla-cache-tier skip_host_uva`` only the anchor layers (those with their
own DSA indexer) keep their fp8_ds_mla cache in HBM. The layers that reuse an
anchor's top-k ("skip" layers) keep theirs in pinned Grace memory. In a decode
step the anchor converts its top-k to this rank's physical slots; right after,
a side stream copies the selected rows of each following skip layer into one
shared set of HBM buffers and remaps the indices to buffer rows, so the skip
layers' FlashMLA calls read HBM. Entries keep their position, multiplicity
and -1, so the kernel reads the same bytes in the same order.

This step's own tokens are written by the skip layer itself, after the copy,
so they cannot be staged: their slots map to reserved rows that the skip
layer's KV write fills (row ``gather_rows() + j`` for token j).

One buffer set serves every group: group g's copy forks from the main stream
after anchor g, when group g-1's skip layers have finished reading.
"""

import os
from functools import cache
from typing import Any

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import tl, triton

TOPK = 2048
MAX_QUERIES = 64  # the most queries a step can stage (one reserved row each)
RESERVED_ROWS = 64
ROW_BYTES = 656
STAGE_BUFFERS = 3  # skip layers per anchor (index_topk_freq - 1)
GATHER_CTAS = 8  # per skip layer


@cache
def limits() -> tuple[int, int]:
    """(queries, index columns) the staging buffers are sized for: a decode
    step's tokens (max_num_seqs x (1 + speculative tokens), at most
    MAX_QUERIES) and, under DCP, the decode index width (each staged row is one
    distinct index entry, so queries x columns rows cover any step). Cached:
    the planner's call at load time (config set) fixes what the stager built
    at the first forward uses; with no config, the worst case."""
    import vllm.envs as envs
    from vllm.config import vllm as vllm_config

    config = vllm_config._current_vllm_config
    if config is None:
        return MAX_QUERIES, TOPK
    spec = config.speculative_config
    queries = config.scheduler_config.max_num_seqs * (
        1 + (spec.num_speculative_tokens if spec is not None else 0)
    )
    columns = TOPK
    if config.parallel_config.decode_context_parallel_size > 1:
        columns = min(envs.VLLM_DCP_SPARSE_DECODE_WIDTH or TOPK, TOPK)
    return min(max(queries, 1), MAX_QUERIES), columns


def gather_rows() -> int:
    queries, columns = limits()
    return queries * columns


def staging_bytes() -> int:
    return STAGE_BUFFERS * (gather_rows() + RESERVED_ROWS) * ROW_BYTES


# A hot shared counter serializes: ~600 winners of one atomic_add took ~20 us
# per group in served decode. Slots are instead marked without atomics, then
# compacted with one counter add per CTA; which row a slot gets does not change
# what any entry reads.


@triton.jit
def _mark_rows_kernel(
    idx_ptr, n, new_ptr, row_of_ptr, T: tl.constexpr, BLOCK: tl.constexpr
):
    """Mark every selected slot that is not this step's (idempotent stores)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(idx_ptr + offs, mask=offs < n, other=-1)
    is_new = s < 0
    for j in tl.static_range(T):
        nj = tl.load(new_ptr + j)
        is_new |= (s == nj) & (nj >= 0)
    want = ~is_new
    tl.store(row_of_ptr + tl.where(want, s, 0).to(tl.int64), -2, mask=want)


@triton.jit
def _compact_rows_kernel(
    row_of_ptr, num_slots, slots_ptr, count_ptr, CHUNK: tl.constexpr
):
    """Give each marked slot of this CTA's range a buffer row."""
    offs = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    marked = tl.load(row_of_ptr + offs, mask=offs < num_slots, other=-1) == -2
    local = tl.cumsum(marked.to(tl.int32), axis=0)
    base = tl.atomic_add(count_ptr, tl.sum(marked.to(tl.int32), axis=0))
    row = base + local - 1
    tl.store(row_of_ptr + offs, row, mask=marked)
    tl.store(slots_ptr + row, offs, mask=marked)


@triton.jit
def _remap_rows_kernel(
    idx_ptr,
    n,
    new_ptr,
    row_of_ptr,
    out_ptr,
    T: tl.constexpr,
    RESERVED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Each entry's buffer row: staged row, reserved row (this step's token),
    or -1."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(idx_ptr + offs, mask=m, other=-1)
    row = tl.load(
        row_of_ptr + tl.where(s >= 0, s, 0).to(tl.int64), mask=m & (s >= 0), other=-1
    )
    for j in tl.static_range(T):
        nj = tl.load(new_ptr + j)
        row = tl.where((s == nj) & (nj >= 0), RESERVED + j, row)
    tl.store(out_ptr + offs, tl.where(s < 0, -1, row), mask=m)


@triton.jit
def _copy_rows(src_ptr, dst_ptr, rows, slots, live, ROW_WORDS, BLOCK_W: tl.constexpr):
    words = tl.arange(0, BLOCK_W)
    mask = live[:, None] & (words < ROW_WORDS)[None, :]
    v = tl.load(src_ptr + slots[:, None] * ROW_WORDS + words[None, :], mask=mask)
    dst = dst_ptr + rows[:, None].to(tl.int64) * ROW_WORDS + words[None, :]
    tl.store(dst, v, mask=mask)


@triton.jit
def _gather_rows_kernel(
    src0,
    src1,
    src2,
    dst_ptr,
    slots_ptr,
    count_ptr,
    row_of_ptr,
    DST_WORDS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """dst[layer] row r = src[layer] row slots[r] for r < count, CTAs striding
    over the rows; layer 0's CTAs also clear the claims for the next group."""
    layer = tl.program_id(1)
    count = tl.load(count_ptr)
    dst = dst_ptr + layer.to(tl.int64) * DST_WORDS
    for start in range(tl.program_id(0) * BLOCK_R, count, tl.num_programs(0) * BLOCK_R):
        rows = start + tl.arange(0, BLOCK_R)
        live = rows < count
        slots = tl.load(slots_ptr + rows, mask=live, other=0).to(tl.int64)
        if layer == 0:
            _copy_rows(src0, dst, rows, slots, live, ROW_WORDS, BLOCK_W)
            tl.store(row_of_ptr + slots, -1, mask=live)
        elif layer == 1:
            _copy_rows(src1, dst, rows, slots, live, ROW_WORDS, BLOCK_W)
        else:
            _copy_rows(src2, dst, rows, slots, live, ROW_WORDS, BLOCK_W)


class SkipKVStager:
    """Per-rank staging state shared by the attention layers on the rank."""

    def __init__(self, device: torch.device, layers: dict[int, Any]):
        self.max_queries, self.columns = limits()
        self.gather_rows = gather_rows()
        rows = self.gather_rows + RESERVED_ROWS
        self.buffers = torch.zeros(
            STAGE_BUFFERS, rows // 64, 64, ROW_BYTES, dtype=torch.uint8, device=device
        )
        self.slots = torch.zeros(self.gather_rows, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.rows = torch.full(
            (self.max_queries, self.columns), -1, dtype=torch.int32, device=device
        )
        self.stream = torch.cuda.Stream(device)
        # anchor cache -> (its skip layers' caches, ready event, group number);
        # skip cache -> buffer position
        self.groups: dict[int, tuple[list[torch.Tensor], torch.cuda.Event, int]]
        self.groups = {}
        self.position: dict[int, int] = {}
        self._ready: torch.cuda.Event | None = None
        self._width = self.columns  # the anchor's index width, set by stage()
        self._reserved = self.gather_rows + torch.arange(
            self.max_queries, dtype=torch.int64, device=device
        )
        self.reserved_map = torch.full_like(self._reserved, -1)
        num_slots = 0
        for index, layer in layers.items():
            if layer.impl._reuses_topk:
                continue
            group: list[torch.Tensor] = []
            while index + 1 + len(group) in layers and (
                layers[index + 1 + len(group)].impl._reuses_topk
            ):
                skip = layers[index + 1 + len(group)].kv_cache
                self.position[skip.data_ptr()] = len(group)
                group.append(skip)
                num_slots = skip.numel() // ROW_BYTES
            assert len(group) <= STAGE_BUFFERS
            if group:
                self.groups[layer.kv_cache.data_ptr()] = (
                    group,
                    torch.cuda.Event(),
                    len(self.groups),
                )
        # slot -> staged buffer row, -2 while marked, -1 otherwise (the last
        # entry is unused padding)
        self.row_of = torch.full((num_slots + 1,), -1, dtype=torch.int32, device=device)
        self.row_of[num_slots] = -3
        # VLLM_SKIP_KV_STATS: per group, a histogram of staged rows in 256-row bins
        self.hist = (
            torch.zeros(len(self.groups), self.gather_rows // 256 + 1, device=device)
            if os.environ.get("VLLM_SKIP_KV_STATS")
            else None
        )

    def stage(
        self, anchor_cache: torch.Tensor, indices: torch.Tensor, new_slots: torch.Tensor
    ) -> None:
        """Anchor, after its FlashMLA launch: start its group's copy."""
        entry = self.groups.get(anchor_cache.data_ptr())
        if entry is None:
            return
        group, ready, number = entry
        t, n = indices.shape[0], indices.numel()
        # every entry may be a distinct row: the buffers must hold them all
        assert t <= self.max_queries and indices.shape[1] <= self.columns, (
            f"skip-KV staging sized for {self.max_queries} x {self.columns}, "
            f"got {tuple(indices.shape)}"
        )
        self._width = indices.shape[1]
        new = new_slots[:t]
        if number == 0:  # this step's tokens are the same for every group
            self.reserved_map[:t] = torch.where(new >= 0, self._reserved[:t], -1)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.count.zero_()
            grid = (triton.cdiv(n, 1024),)
            _mark_rows_kernel[grid](indices, n, new, self.row_of, T=t, BLOCK=1024)
            num_slots = self.row_of.numel() - 1
            _compact_rows_kernel[(triton.cdiv(num_slots, 4096),)](
                self.row_of, num_slots, self.slots, self.count, CHUNK=4096
            )
            _remap_rows_kernel[grid](
                indices,
                n,
                new,
                self.row_of,
                self.rows,
                T=t,
                RESERVED=self.gather_rows,
                BLOCK=1024,
            )
            srcs = [c.view(torch.int32) for c in group]
            srcs += [srcs[0]] * (STAGE_BUFFERS - len(srcs))
            words = ROW_BYTES // 4
            _gather_rows_kernel[(GATHER_CTAS, len(group))](
                *srcs,
                self.buffers.view(torch.int32),
                self.slots,
                self.count,
                self.row_of,
                DST_WORDS=(self.gather_rows + RESERVED_ROWS) * words,
                ROW_WORDS=words,
                BLOCK_R=32,
                BLOCK_W=triton.next_power_of_2(words),
            )
            if self.hist is not None:
                self.hist[number].index_add_(
                    0, (self.count // 256).long(), self.count.new_ones(1).float()
                )
            ready.record(self.stream)
        self._ready = ready

    def staged(
        self, cache: torch.Tensor, num_queries: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Skip layer: its staged buffer and remapped indices, else None."""
        p = self.position.get(cache.data_ptr())
        if p is None:
            return None
        assert self._ready is not None
        torch.cuda.current_stream().wait_event(self._ready)
        rows = self.rows.view(-1)[: num_queries * self._width]
        return self.buffers[p], rows.view(num_queries, self._width)

    def write_new(
        self,
        cache: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """Skip layer's KV write: this step's tokens into its reserved rows
        (the mapping is built by the anchor's ``stage``)."""
        p = self.position.get(cache.data_ptr())
        if p is None:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            self.buffers[p],
            self.reserved_map[: slot_mapping.numel()],
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )


_STAGER: SkipKVStager | None = None


def get_stager(device: torch.device) -> SkipKVStager:
    """The rank's stager, built from the bound attention layers on first use."""
    global _STAGER
    if _STAGER is None:
        layers = {
            extract_layer_index(name): layer
            for name, layer in get_forward_context().no_compile_layers.items()
            if hasattr(getattr(layer, "impl", None), "_reuses_topk")
        }
        _STAGER = SkipKVStager(device, layers)
    return _STAGER
```
## vllm/v1/attention/backends/mla/flashmla_sparse.py excerpts (unchanged): forward_mqa staging + do_kv_cache_update
```python
        Used when use_mixed_batch is True.
        """
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). Under DCP, keep only this rank's KV shard: owned
        # slots are compacted to a contiguous prefix and the tail stays -1,
        # which the kernel skips; the LSE-based cross-rank combine restores
        # the full softmax.
        if (
            attn_metadata.dcp_prefill_local_len > 0
            and q.shape[0] == attn_metadata.num_actual_tokens
            and not torch.cuda.is_current_stream_capturing()
        ):
            return self._forward_dcp_sparse_prefill(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        valid_counts: torch.Tensor | None = None
        sched: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.dcp_world_size > 1:
            topk_indices, valid_counts, sched = self._dcp_converted_indices(
                topk_indices, attn_metadata, decode_sched=self._dcp_decode_sched(q)
            )
        else:
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )
        stager = (
            skip_kv_stage.get_stager(q.device)
            if self._skip_kv
            and attn_metadata.num_prefills == 0
            and q.shape[0] <= skip_kv_stage.limits()[0]
            else None
        )
        if (
            stager is not None
            and self._reuses_topk
            and (staged := stager.staged(kv_c_and_k_pe_cache, q.shape[0]))
        ):
            kv_c_and_k_pe_cache, topk_indices = staged

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        if sched is not None:
            _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
                q.unsqueeze(0),
                kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
                topk_indices.unsqueeze(0),
                None,
                None,
                *sched,
                None,
                None,
                None,
                512,
                self.softmax_scale,
                None,
            )
        else:
            _attn_out, _lse = self._fp8_flash_mla_kernel(
                q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
                kernel_metadata=fp8_metadata,
            )

        if stager is not None and not self._reuses_topk:
            # after the anchor's own attention, so the copy does not slow it
            stager.stage(kv_c_and_k_pe_cache, topk_indices, attn_metadata.slot_mapping)

        # Output is (1, T, H, D_v) -> (T, H, D_v); lse is (1, H, T) -> (T, H)
        lse = _lse.squeeze(0).transpose(0, 1)
        if valid_counts is not None and not self.dcp_combine_drops_inf_lse:
            lse = mask_empty_dcp_lse(lse, valid_counts)
        return _attn_out.squeeze(0), lse

    def _forward_dcp_sparse_prefill(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, valid_counts, _ = self._dcp_converted_indices(
            topk_indices, attn_metadata, prefill=True
        )
        # the combine's registered buffer, when it has one: no staging copy
        one_shot = _one_shot(get_dcp_group())
        out = (
            None
            if one_shot is None
            else one_shot.prefill_out(*q.shape[:2], self.kv_lora_rank, q.dtype)
        )
        return dcp_sparse_prefill(
            q,
            kv_c_and_k_pe_cache,
            attn_metadata.block_table[:1],
            attn_metadata.dcp_prefill_local_len,
            indices,
            valid_counts,
            self.softmax_scale,
            mask_empty_lse=not self.dcp_combine_drops_inf_lse,
            out=out,
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
        if (
            self._skip_kv
            and self._reuses_topk
            and slot_mapping.numel() <= skip_kv_stage.limits()[0]
        ):
            skip_kv_stage.get_stager(kv_cache.device).write_new(
                kv_cache, kv_c_normed, k_pe, slot_mapping, k_scale
            )

    def _dcp_decode_sched(self, q: torch.Tensor) -> bool:
```
## vllm/model_executor/layers/attention/mla_attention.py: unified_mla_kv_cache_update (unchanged)
```python
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))


def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            kv_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


direct_register_custom_op(
    op_name="unified_mla_kv_cache_update",
    op_func=unified_mla_kv_cache_update,
    fake_impl=unified_mla_kv_cache_update_fake,
)


@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
```
## vllm/model_executor/layers/attention/attention.py get_attention_context (unchanged)
```python
def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: AttentionMetadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


def unified_kv_cache_update(
```
## vllm/model_executor/layers/rotary_embedding/base.py RotaryEmbedding forward_static (unchanged)
```python
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = ApplyRotaryEmb.forward_static(
            query_rot,
            cos,
            sin,
            is_neox_style,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            key_rot = ApplyRotaryEmb.forward_static(
                key_rot,
                cos,
                sin,
                is_neox_style,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

```
## vllm/v1/attention/backend.py MLA do_kv_cache_update (unchanged)
```python
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
```
