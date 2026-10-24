You are reviewing a CUDA-graph multi-stream change in a vLLM fork. Do NOT run any commands (your sandbox cannot run here); reason only from the material below. Be concrete: cite file:line and describe exact interleavings.

## Context
GLM-5.3 (DeepSeek-V3.2-style DSA sparse MLA), TP4 x DCP4 (decode context parallel, 4 ranks share the KV), decode verify step M=16 tokens captured as one FULL CUDA graph (the V2 model runner captures FULL graphs as a NONE-mode forward with attention metadata set). 78 layers; "anchor" layers run the DSA indexer (wk/weights proj, k_norm, wq_b, q rope/fp8 quant, indexer K-cache write, paged MQA logits, cooperative radix top-k, DCP merge = pack candidates -> one-shot all-gather over the DCP group -> CuteDSL StableTopK) and write top-k into a shared topk_indices_buffer that the anchor's own sparse FlashMLA and the following 3 "shared" layers read. Previously everything ran serially on the main stream.

Change (env VLLM_ANCHOR_INDEXER_FORK=1, only inside FULL capture): fork the indexer's k projection at the layer entry onto a side stream (overlaps qkv_a GEMM + fused qkv glue), fork the rest of the indexer after q_c is available (overlaps q_b GEMM, q RoPE, the layer's KV write, W_UK bmm, DCP q all-gather), and join (a) the anchor hot-cache's own side stream in start_mqa (it converts/maps top-k to hot-cache slots and copies rows from Grace memory) and (b) the main stream in forward_mqa before reading topk_indices_buffer. Math must be bitwise unchanged in intent (same ops); the indexer inside the custom op runs eagerly rather than through Inductor.

Commit 2 fixes a hazard I found: the indexer's DCP candidate all-gather and the main stream's q all_gather_cat are both one-shot kernels on the custom all-reduce's Signal (barrier_at_start reads self_sg->_flag[blockIdx.x] + 1, writes it to peers, spins `!= flag`, then stores _flag). Run concurrently they race on _flag -> hang or early barrier. Now the main q gather waits on an event recorded right after the indexer's all-gather (StableTopK still overlaps the q gather).

## What I want from you
1. Any remaining cross-stream hazard: shared buffers (workspace manager returns views at offset 0 of ONE shared buffer — see excerpt), communicator state, cuBLAS workspaces, caching-allocator reuse across streams inside capture (record_stream usage), the topk buffer itself (it is also written by the previous anchor and read by the 3 shared layers in between — is there a WAR hazard where the side indexer of anchor N overwrites topk_indices_buffer while the main stream's shared layer N-1 (or its skip-KV side branch) still reads it?), events reused across replays, collective ordering consistency across the 4 ranks, graph-buffer registration order for the custom all-reduce.
2. Correctness of the host-side state machine (_pending/_recording/_gathered, per-layer events, asserts) across: eager steps, piecewise-graph capture, FULL capture, warmup runs inside capture contexts, dummy/profile runs, the MTP drafter (also DSA, has an indexer), and torch.compile/Inductor (the ops are opaque custom ops with mutates_args=["topk_indices_buffer"] and no tensor outputs — can Inductor reorder them relative to the main-stream ops that do not touch topk_indices_buffer, e.g. move fork_q after the q gather, or DCE them?).
3. Whether the test meaningfully guards the behaviors.
4. Anything that would make the measured speedup misleading.
Rank findings by severity; say "no issue" explicitly for checked areas.

## Diff (23d2796ed5..c6c01b1989)
```diff
diff --git a/tests/model_executor/test_indexer_fork.py b/tests/model_executor/test_indexer_fork.py
new file mode 100644
index 0000000000..1faffea27b
--- /dev/null
+++ b/tests/model_executor/test_indexer_fork.py
@@ -0,0 +1,92 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""indexer_fork: the forked indexer's top-k reaches its consumer inside a
+captured CUDA graph (fork at the layer entry and after q_c, join before the
+read, and the main stream's collective after the indexer's), on replays with
+new inputs."""
+
+import pytest
+import torch
+
+from vllm.model_executor.layers import indexer_fork
+
+pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
+
+
+class _Indexer(torch.nn.Module):
+    """Stand-in: a slow k projection, a "collective" writing ``comm``, and a
+    top-k written into the buffer."""
+
+    def __init__(self, buf: torch.Tensor, d: int):
+        super().__init__()
+        g = torch.Generator(device="cuda").manual_seed(0)
+        self.wk = torch.randn(d, d, device="cuda", generator=g)
+        self.wq = torch.randn(d, d, device="cuda", generator=g)
+        self.buf = buf
+        self.comm = torch.zeros_like(buf, dtype=torch.float32)
+
+    def project_k(self, h):
+        k = h
+        for _ in range(64):  # long enough that a missing join reads stale data
+            k = torch.tanh(k @ self.wk)
+        return k, h.sum(-1, keepdim=True)
+
+    def forward(self, h, q_c, positions, rope, kw):
+        k, w = kw
+        s = (q_c @ self.wq) * k + w + positions[:, None]
+        self.comm.copy_(s[:, : self.comm.shape[1]])
+        indexer_fork.gathered()
+        self.buf.copy_(s.topk(self.buf.shape[1], dim=-1).indices.int())
+
+
+def _layer(name: str, buf: torch.Tensor, d: int):
+    w = torch.nn.Module()
+    w.indexer = _Indexer(buf, d)
+    w.indexer_rope_emb = None
+    indexer_fork.register(name, w)
+    return w
+
+
+def _step(h, q_c, pos, buf, comm, name):
+    torch.ops.vllm.mla_indexer_fork_k(h, buf, name)
+    q_c = q_c * 2  # main-stream work between the forks
+    torch.ops.vllm.mla_indexer_fork_q(h, q_c, pos, buf, name)
+    indexer_fork.join_gather(torch.cuda.current_stream())
+    c = comm.sum(-1)  # the main stream's collective
+    out = torch.tanh(h @ h.T).sum()  # main-stream work the fork overlaps
+    indexer_fork.join(torch.cuda.current_stream(), clear=True)
+    return buf.float().sum(-1) + out + c
+
+
+def test_forked_topk_joined_in_graph(monkeypatch):
+    t, d, k = 16, 512, 8
+    buf = torch.zeros(t, k, dtype=torch.int32, device="cuda")
+    comm = _layer("l0", buf, d).indexer.comm
+    h, q_c = (torch.randn(t, d, device="cuda") for _ in range(2))
+    pos = torch.arange(t, device="cuda", dtype=torch.float32)
+
+    monkeypatch.setattr(indexer_fork, "_forking", lambda: False)
+    want = []
+    for _ in range(3):
+        h.normal_(), q_c.normal_()
+        want.append(
+            (h.clone(), q_c.clone(), _step(h, q_c, pos, buf, comm, "l0").clone())
+        )
+
+    monkeypatch.setattr(
+        indexer_fork, "_forking", torch.cuda.is_current_stream_capturing
+    )
+    side = torch.cuda.Stream()
+    side.wait_stream(torch.cuda.current_stream())
+    with torch.cuda.stream(side):
+        _step(h, q_c, pos, buf, comm, "l0")  # warm up off-capture (no fork)
+    torch.cuda.current_stream().wait_stream(side)
+    g = torch.cuda.CUDAGraph()
+    with torch.cuda.graph(g):
+        got = _step(h, q_c, pos, buf, comm, "l0")
+    assert indexer_fork._pending is None and indexer_fork._gathered is None
+    for h0, q0, ref in want:
+        h.copy_(h0), q_c.copy_(q0), buf.zero_(), comm.zero_()
+        g.replay()
+        torch.accelerator.synchronize()
+        torch.testing.assert_close(got, ref, rtol=0, atol=0)
diff --git a/vllm/envs.py b/vllm/envs.py
index 9fd9c19c74..0f7347c309 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -289,6 +289,7 @@ if TYPE_CHECKING:
     VLLM_ANCHOR_HOT_CACHE_CHECK: bool = False
     VLLM_DCP_DECODE_SCHED: bool = True
     VLLM_MLA_QKV_GLUE: bool = False
+    VLLM_ANCHOR_INDEXER_FORK: bool = False
     VLLM_SCHED_INTERLEAVE_PREFILL: int = 0
     VLLM_SCHED_INTERLEAVE_SRPT: bool = False
     VLLM_TIERED_MOE_KV_POOL_SEQS: int = 0
@@ -2056,6 +2057,11 @@ environment_variables: dict[str, Callable[[], Any]] = {
     # the k_pe RoPE and the layer's KV writes between fused_qkv_a and q_b in
     # one kernel (vllm/model_executor/layers/mla_glue).
     "VLLM_MLA_QKV_GLUE": lambda: bool(int(os.getenv("VLLM_MLA_QKV_GLUE", "0"))),
+    # Anchors' DSA indexer on a side stream inside FULL CUDA graphs, joined
+    # where the top-k is read (vllm/model_executor/layers/indexer_fork.py)
+    "VLLM_ANCHOR_INDEXER_FORK": lambda: bool(
+        int(os.getenv("VLLM_ANCHOR_INDEXER_FORK", "0"))
+    ),
     # Scheduler: never mix prefill and decode tokens in one step. With N > 0,
     # a step is either decode-only or one request's prefill chunk, and N
     # decode-only steps run between prefill chunks while requests decode.
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index 9655610e06..4d5f086b53 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -216,6 +216,7 @@ from vllm.distributed.parallel_state import (
 from vllm.forward_context import ForwardContext, get_forward_context
 from vllm.logger import init_logger
 from vllm.model_executor.custom_op import CustomOp
+from vllm.model_executor.layers import indexer_fork
 from vllm.model_executor.layers.attention.attention import (
     _init_kv_cache_quant,
     get_attention_context,
@@ -866,6 +867,7 @@ class MLAAttention(nn.Module, AttentionLayerBase):
                 mqa_q = get_dcp_group().all_gather(q_head_major, dim=0).transpose(0, 1)
             elif self.impl.dcp_world_size > 1:
                 # all-gather in the head dim; a (nope, pe) pair -> (B, N, L + P)
+                indexer_fork.join_gather(torch.cuda.current_stream())
                 mqa_q = cp_all_gather_q(mqa_q, get_dcp_group())
 
             # call decode attn
diff --git a/vllm/model_executor/layers/indexer_fork.py b/vllm/model_executor/layers/indexer_fork.py
new file mode 100644
index 0000000000..973940a3a1
--- /dev/null
+++ b/vllm/model_executor/layers/indexer_fork.py
@@ -0,0 +1,150 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The DSA indexer of an anchor layer on a side stream (VLLM_ANCHOR_INDEXER_FORK).
+
+Inside a FULL CUDA graph capture the indexer's key projection (wk, k_norm)
+forks at the layer's entry, alongside qkv_a. The rest of the indexer forks
+after q_a's norm: wq_b, the q RoPE/quant, the indexer cache write, logits
+and the top-k. Meanwhile the main stream runs q_b, the q RoPE, the layer's
+KV write, W_UK and the DCP query gather. The top-k's consumers join the side
+stream: the anchor hot cache's convert/map stream in start_mqa, and the main
+stream in forward_mqa. The indexer's DCP candidate all-gather and the main
+stream's query all-gather are one-shot collectives sharing the custom
+all-reduce's barrier flags, so the query gather waits for the indexer's.
+Piecewise graphs and eager steps run the same ops inline. The math is
+unchanged, but these ops run eagerly rather than through Inductor.
+"""
+
+import torch
+
+from vllm.forward_context import get_forward_context
+from vllm.utils.torch_utils import (
+    LayerNameType,
+    _resolve_layer_name,
+    direct_register_custom_op,
+)
+
+_layers: dict[str, torch.nn.Module] = {}
+_stream: torch.cuda.Stream | None = None
+_pending: torch.cuda.Event | None = None
+_recording: torch.cuda.Event | None = None
+_gathered: torch.cuda.Event | None = None
+
+
+def register(layer_name: str, wrapper: torch.nn.Module) -> None:
+    _layers[layer_name] = wrapper
+    wrapper._indexer_kw = None
+    wrapper._indexer_forked = False
+    wrapper._indexer_done = torch.cuda.Event()
+    wrapper._indexer_gathered = torch.cuda.Event()
+
+
+def _side() -> torch.cuda.Stream:
+    global _stream
+    if _stream is None:
+        _stream = torch.cuda.Stream()
+    return _stream
+
+
+def _forking() -> bool:
+    """Inside a FULL graph capture: the V2 runner captures FULL graphs as a
+    NONE-mode forward with attention metadata (as mla_glue.writes_kv)."""
+    if not torch.cuda.is_current_stream_capturing():
+        return False
+    fc = get_forward_context()
+    return fc.attn_metadata is not None and bool(fc.slot_mapping)
+
+
+def join(stream: torch.cuda.Stream, clear: bool = False) -> None:
+    """Make ``stream`` wait for the forked indexer's top-k, if one is pending."""
+    global _pending, _gathered
+    if _pending is not None:
+        stream.wait_event(_pending)
+        if clear:
+            _pending = _gathered = None
+
+
+def gathered() -> None:
+    """Called after the indexer's DCP all-gather; marks it on the side stream."""
+    global _gathered
+    if _recording is not None:
+        _recording.record()
+        _gathered = _recording
+
+
+def join_gather(stream: torch.cuda.Stream) -> None:
+    """Order a one-shot collective on ``stream`` after the forked indexer's."""
+    global _gathered
+    if _gathered is not None:
+        stream.wait_event(_gathered)
+        _gathered = None
+
+
+def _fork_k(
+    hidden_states: torch.Tensor,
+    topk_indices_buffer: torch.Tensor,
+    layer_name: LayerNameType,
+) -> None:
+    w = _layers[_resolve_layer_name(layer_name)]
+    w._indexer_forked = _forking()
+    if not w._indexer_forked:
+        w._indexer_kw = w.indexer.project_k(hidden_states)
+        return
+    main, side = torch.cuda.current_stream(), _side()
+    side.wait_stream(main)
+    hidden_states.record_stream(side)
+    with torch.cuda.stream(side):
+        w._indexer_kw = w.indexer.project_k(hidden_states)
+
+
+def _fork_q(
+    hidden_states: torch.Tensor,
+    q_c: torch.Tensor,
+    positions: torch.Tensor,
+    topk_indices_buffer: torch.Tensor,
+    layer_name: LayerNameType,
+) -> None:
+    global _pending, _recording
+    w = _layers[_resolve_layer_name(layer_name)]
+    kw, w._indexer_kw = w._indexer_kw, None
+    assert kw is not None, "mla_indexer_fork_q without mla_indexer_fork_k"
+    if not w._indexer_forked:
+        w.indexer(hidden_states, q_c, positions, w.indexer_rope_emb, kw)
+        return
+    assert _pending is None, "previous anchor's indexer was never joined"
+    main, side = torch.cuda.current_stream(), _side()
+    side.wait_stream(main)
+    for t in (hidden_states, q_c, positions):
+        t.record_stream(side)
+    _recording = w._indexer_gathered
+    try:
+        with torch.cuda.stream(side):
+            w.indexer(hidden_states, q_c, positions, w.indexer_rope_emb, kw)
+            w._indexer_done.record(side)
+    finally:
+        _recording = None
+    _pending = w._indexer_done
+
+
+def _fork_k_fake(hidden_states, topk_indices_buffer, layer_name) -> None:
+    return None
+
+
+def _fork_q_fake(
+    hidden_states, q_c, positions, topk_indices_buffer, layer_name
+) -> None:
+    return None
+
+
+direct_register_custom_op(
+    op_name="mla_indexer_fork_k",
+    op_func=_fork_k,
+    mutates_args=["topk_indices_buffer"],
+    fake_impl=_fork_k_fake,
+)
+direct_register_custom_op(
+    op_name="mla_indexer_fork_q",
+    op_func=_fork_q,
+    mutates_args=["topk_indices_buffer"],
+    fake_impl=_fork_q_fake,
+)
diff --git a/vllm/model_executor/layers/mla.py b/vllm/model_executor/layers/mla.py
index 154cfda935..088abb18e1 100644
--- a/vllm/model_executor/layers/mla.py
+++ b/vllm/model_executor/layers/mla.py
@@ -4,6 +4,7 @@ from dataclasses import dataclass
 
 import torch
 
+import vllm.envs as envs
 from vllm.config import CacheConfig
 from vllm.model_executor.custom_op import PluggableLayer
 from vllm.model_executor.layers.attention import MLAAttention
@@ -130,6 +131,29 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
         self.qkv_glue = mla_glue.supported(self)
         self.mla_attn.qkv_glue = self.qkv_glue
 
+        self.indexer_fork = (
+            envs.VLLM_ANCHOR_INDEXER_FORK
+            and self.indexer is not None
+            and self.is_sparse
+            and not self.skip_topk
+        )
+        if self.indexer_fork:
+            from vllm.model_executor.layers import indexer_fork
+
+            indexer_fork.register(self.mla_attn.layer_name, self)
+
+    def _fork_indexer_q(
+        self, hidden_states: torch.Tensor, q_c: torch.Tensor, positions: torch.Tensor
+    ) -> None:
+        if self.indexer_fork:
+            torch.ops.vllm.mla_indexer_fork_q(
+                hidden_states,
+                q_c,
+                positions,
+                self.topk_indices_buffer,
+                _encode_layer_name(self.mla_attn.layer_name),
+            )
+
     def forward(
         self,
         positions: torch.Tensor,
@@ -138,6 +162,12 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
     ) -> torch.Tensor:
         q_c = None
         kv_lora = None
+        if self.indexer_fork:
+            torch.ops.vllm.mla_indexer_fork_k(
+                hidden_states,
+                self.topk_indices_buffer,
+                _encode_layer_name(self.mla_attn.layer_name),
+            )
 
         if self.qkv_glue:
             assert self.fused_qkv_a_proj is not None
@@ -152,6 +182,7 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
                 self.kv_a_layernorm.variance_epsilon,
                 _encode_layer_name(self.mla_attn.layer_name),
             )
+            self._fork_indexer_q(hidden_states, q_c, positions)
             q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
             k_pe = k_pe.unsqueeze(1)
             q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
@@ -174,6 +205,7 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
                 dim=-1,
             )
             q_c = self.q_a_layernorm(q_c)
+            self._fork_indexer_q(hidden_states, q_c, positions)
             q = self.q_b_proj(q_c)[0]
         else:
             assert self.kv_a_proj_with_mqa is not None, (
@@ -204,7 +236,12 @@ class MultiHeadLatentAttentionWrapper(PluggableLayer):
         if self.cold_prefetch_layer is not None:
             torch.ops.vllm.tiered_moe_cold_join(q)
 
-        if self.indexer and self.is_sparse and not self.skip_topk:
+        if (
+            self.indexer
+            and self.is_sparse
+            and not self.skip_topk
+            and not self.indexer_fork
+        ):
             self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
 
         if llama_4_scaling is not None:
diff --git a/vllm/model_executor/layers/sparse_attn_indexer.py b/vllm/model_executor/layers/sparse_attn_indexer.py
index ceb52e5d32..fa0aef636b 100644
--- a/vllm/model_executor/layers/sparse_attn_indexer.py
+++ b/vllm/model_executor/layers/sparse_attn_indexer.py
@@ -13,6 +13,7 @@ from vllm.distributed import get_dcp_group
 from vllm.forward_context import get_forward_context
 from vllm.logger import init_logger
 from vllm.model_executor.custom_op import CustomOp
+from vllm.model_executor.layers import indexer_fork
 from vllm.model_executor.layers.quantization.utils.quant_utils import (
     get_fp8_min_max,
 )
@@ -118,6 +119,7 @@ def _merge_dcp_topk_global(
         row_starts,
     )
     gathered = get_dcp_group().all_gather(packed, dim=1)
+    indexer_fork.gathered()
     stable_topk_from_gathered_candidates_cutedsl(
         gathered, topk_tokens, out=topk_indices
     )
diff --git a/vllm/model_executor/models/deepseek_v2.py b/vllm/model_executor/models/deepseek_v2.py
index 71c3df2fdf..4653888bea 100644
--- a/vllm/model_executor/models/deepseek_v2.py
+++ b/vllm/model_executor/models/deepseek_v2.py
@@ -738,9 +738,22 @@ class Indexer(nn.Module):
             self.index_trace_capture_fn(topk_indices)
         return topk_indices
 
+    def project_k(
+        self, hidden_states: torch.Tensor
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        """The fused wk + weights_proj GEMM and k_norm: (normed k, raw weights)."""
+        kw, _ = self.wk_weights_proj(hidden_states)
+        return self.k_norm(kw[:, : self.head_dim]), kw[:, self.head_dim :]
+
     def forward(
-        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
+        self,
+        hidden_states: torch.Tensor,
+        qr: torch.Tensor,
+        positions,
+        rotary_emb,
+        k_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
     ) -> torch.Tensor:
+        k, weights = self.project_k(hidden_states) if k_weights is None else k_weights
         q, _ = self.wq_b(qr)
         q = q.view(-1, self.n_head, self.head_dim)
 
@@ -751,23 +764,10 @@ class Indexer(nn.Module):
             # On ROCm, this is only valid for kernels used as custom ops.
             # In pytorch-native rope for inductor fusion, rotated q/k tensors
             # are not mutated inplace but returned as new tensors.
-            # Fused wk + weights_proj: one GEMM, then split
-            kw, _ = self.wk_weights_proj(hidden_states)
-            k = kw[:, : self.head_dim]
-            weights = kw[:, self.head_dim :]
-
-            k = self.k_norm(k)
-
             rotary_emb(
                 positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
             )
         elif self.use_fused_indexer_q and q.dtype == torch.bfloat16:
-            # fused wk + weights_proj: one GEMM, then split
-            kw, _ = self.wk_weights_proj(hidden_states)
-            k = kw[:, : self.head_dim]
-            weights = kw[:, self.head_dim :]
-
-            k = self.k_norm(k)
             k_pe, k_nope = torch.split(
                 k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
             )
@@ -796,12 +796,6 @@ class Indexer(nn.Module):
             q_pe, q_nope = torch.split(
                 q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
             )
-            # Fused wk + weights_proj: one GEMM, then split
-            kw, _ = self.wk_weights_proj(hidden_states)
-            k = kw[:, : self.head_dim]
-            weights = kw[:, self.head_dim :]
-
-            k = self.k_norm(k)
             k_pe, k_nope = torch.split(
                 k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
             )
diff --git a/vllm/v1/attention/backends/mla/flashmla_sparse.py b/vllm/v1/attention/backends/mla/flashmla_sparse.py
index 0e2726beb0..180ed0cd2c 100644
--- a/vllm/v1/attention/backends/mla/flashmla_sparse.py
+++ b/vllm/v1/attention/backends/mla/flashmla_sparse.py
@@ -11,6 +11,7 @@ from vllm.config import VllmConfig, get_current_vllm_config
 from vllm.config.cache import CacheDType
 from vllm.distributed.parallel_state import get_dcp_group
 from vllm.logger import init_logger
+from vllm.model_executor.layers import indexer_fork
 from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
 from vllm.model_executor.layers.attention.sparse_mla_attention import (
     SparseMLACommonImpl,
@@ -1110,6 +1111,7 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         main = torch.cuda.current_stream()
         side = anchor_hot_cache.stream()
         side.wait_stream(main)
+        indexer_fork.join(side)
         new_slots = attn_metadata.slot_mapping[:num_tokens]
         with torch.cuda.stream(side):
             indices, valid_counts, sched = self._converted_indices(
@@ -1313,6 +1315,7 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
 
         # Get topk indices
         assert self.topk_indices_buffer is not None
+        indexer_fork.join(torch.cuda.current_stream(), clear=True)
         topk_indices = self.topk_indices_buffer[:num_actual_toks]
 
         use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
```

## Excerpts (unchanged code, worktree at c6c01b1989)

### vllm/v1/attention/backends/mla/flashmla_sparse.py:1096-1130
```
1096:     def start_mqa(
1097:         self,
1098:         kv_cache: torch.Tensor,
1099:         attn_metadata: FlashMLASparseMetadata,
1100:         num_tokens: int,
1101:     ) -> None:
1102:         """Start of the decode attention, before the query absorb and its
1103:         all-gather: convert and map this anchor's top-k on the hot cache's
1104:         stream, so the copy from Grace overlaps them."""
1105:         if not envs.VLLM_ANCHOR_HOT_CACHE_FORK:
1106:             return
1107:         hot = self._hot_cache(kv_cache, attn_metadata)
1108:         if hot is None or not attn_metadata.fp8_use_mixed_batch:
1109:             return
1110:         assert self.topk_indices_buffer is not None
1111:         main = torch.cuda.current_stream()
1112:         side = anchor_hot_cache.stream()
1113:         side.wait_stream(main)
1114:         indexer_fork.join(side)
1115:         new_slots = attn_metadata.slot_mapping[:num_tokens]
1116:         with torch.cuda.stream(side):
1117:             indices, valid_counts, sched = self._converted_indices(
1118:                 self.topk_indices_buffer[:num_tokens],
1119:                 attn_metadata,
1120:                 self._dcp_decode_sched(self.num_heads * self.dcp_world_size),
1121:             )
1122:             rows = hot.map(indices, new_slots)
1123:             hot.ready.record(side)
1124:         # allocated on the side stream, read on main
1125:         for t in (indices, valid_counts, *(sched or ())):
1126:             if t is not None:
1127:                 t.record_stream(main)
1128:         new_slots.record_stream(side)
1129:         self._hot_pending = (indices, valid_counts, sched, rows, hot.ready)
1130: 
```

### vllm/v1/attention/backends/mla/flashmla_sparse.py:1295-1330
```
1295:     def forward_mqa(
1296:         self,
1297:         q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
1298:         kv_c_and_k_pe_cache: torch.Tensor,
1299:         attn_metadata: FlashMLASparseMetadata,
1300:         layer: AttentionLayer,
1301:     ) -> tuple[torch.Tensor, torch.Tensor | None]:
1302:         # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
1303:         # MQA 576/512 approach for both prefill and decode
1304: 
1305:         # Concatenate q if it's a tuple (ql_nope, q_pe)
1306:         if isinstance(q, tuple):
1307:             ql_nope, q_pe = q
1308:             q_concat_buffer, *_ = current_workspace_manager().get_simultaneous(
1309:                 *self.workspace_specs
1310:             )
1311:             q = q_concat_buffer[: ql_nope.shape[0]]
1312:             ops.concat_mla_q(ql_nope, q_pe, q)
1313: 
1314:         num_actual_toks = q.shape[0]
1315: 
1316:         # Get topk indices
1317:         assert self.topk_indices_buffer is not None
1318:         indexer_fork.join(torch.cuda.current_stream(), clear=True)
1319:         topk_indices = self.topk_indices_buffer[:num_actual_toks]
1320: 
1321:         use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"
1322: 
1323:         if self.dcp_world_size > 1 and not (
1324:             use_fp8_cache and attn_metadata.fp8_use_mixed_batch
1325:         ):
1326:             raise NotImplementedError(
1327:                 "FlashMLA sparse DCP requires the fp8_ds_mla mixed-batch path "
1328:                 "(num_heads < MIN_HEADS_FOR_BF16_PREFILL)"
1329:             )
1330: 
```

### vllm/model_executor/layers/attention/mla_attention.py:774-782
```
774:             )
775: 
776:         if num_mqa_tokens > 0:
777:             start_mqa = getattr(self.impl, "start_mqa", None)
778:             if start_mqa is not None:
779:                 start_mqa(kv_cache, attn_metadata, num_mqa_tokens)
780:             mqa_q = q[:num_mqa_tokens]
781:             mqa_output_slice = output[:num_mqa_tokens]
782: 
```

### vllm/model_executor/layers/attention/mla_attention.py:862-878
```
862:                 )
863:             else:
864:                 mqa_q = (mqa_ql_nope, mqa_q_pe)
865:             if q_head_major is not None:
866:                 q_head_major[..., self.kv_lora_rank :].copy_(mqa_q_pe.transpose(0, 1))
867:                 mqa_q = get_dcp_group().all_gather(q_head_major, dim=0).transpose(0, 1)
868:             elif self.impl.dcp_world_size > 1:
869:                 # all-gather in the head dim; a (nope, pe) pair -> (B, N, L + P)
870:                 indexer_fork.join_gather(torch.cuda.current_stream())
871:                 mqa_q = cp_all_gather_q(mqa_q, get_dcp_group())
872: 
873:             # call decode attn
874:             if not self.impl.is_sparse:
875:                 assert attn_metadata.decode is not None
876:             attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]
877:             o_proj = getattr(self, "l2_prefetch_o_proj", None)
878:             if envs.VLLM_DENSE_L2_PREFETCH and o_proj is not None:
```

### vllm/model_executor/layers/sparse_attn_indexer.py:296-400
```
296: @eager_break_during_capture
297: def sparse_attn_indexer(
298:     hidden_states: torch.Tensor,
299:     k_cache_prefix: LayerNameType,
300:     kv_cache: torch.Tensor,
301:     q_quant: torch.Tensor,
302:     q_scale: torch.Tensor | None,
303:     k: torch.Tensor,
304:     weights: torch.Tensor,
305:     quant_block_size: int,
306:     scale_fmt: str | None,
307:     topk_tokens: int,
308:     head_dim: int,
309:     max_model_len: int,
310:     total_seq_lens: int,
311:     topk_indices_buffer: torch.Tensor,
312:     skip_k_cache_insert: bool,
313:     use_fp4_cache: bool = False,
314:     dcp_rank: int = 0,
315:     dcp_world_size: int = 1,
316:     cp_kv_cache_interleave_size: int = 1,
317:     skip_topk_buffer_clear: bool = False,
318: ) -> torch.Tensor:
319:     # careful! this will be None in dummy run
320:     attn_metadata = get_forward_context().attn_metadata
321:     fp8_dtype = current_platform.fp8_dtype()
322:     k_cache_prefix = _resolve_layer_name(k_cache_prefix)
323: 
324:     # assert isinstance(attn_metadata, dict)
325:     if not isinstance(attn_metadata, dict):
326:         # Reserve workspace for indexer during profiling run
327:         values_spec, scales_spec = _gather_workspace_shapes(
328:             total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
329:         )
330:         current_workspace_manager().get_simultaneous(
331:             values_spec,
332:             scales_spec,
333:             ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
334:         )
335: 
336:         # Dummy allocation to simulate for peak logits tensor memory during inference.
337:         # FP8 elements so elements == bytes
338:         max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
339:         _ = torch.empty(
340:             max_logits_elems, dtype=torch.uint8, device=hidden_states.device
341:         )
342: 
343:         return sparse_attn_indexer_fake(
344:             hidden_states,
345:             k_cache_prefix,
346:             kv_cache,
347:             q_quant,
348:             q_scale,
349:             k,
350:             weights,
351:             quant_block_size,
352:             scale_fmt,
353:             topk_tokens,
354:             head_dim,
355:             max_model_len,
356:             total_seq_lens,
357:             topk_indices_buffer,
358:             skip_k_cache_insert,
359:             use_fp4_cache,
360:         )
361:     attn_metadata_narrowed = attn_metadata[k_cache_prefix]
362:     assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
363:     slot_mapping = attn_metadata_narrowed.slot_mapping
364:     has_decode = attn_metadata_narrowed.num_decodes > 0
365:     has_prefill = attn_metadata_narrowed.num_prefills > 0
366:     num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
367: 
368:     # q_scale is required iff the FP4 cache path is enabled; the FP8 path
369:     # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
370:     if use_fp4_cache:
371:         assert q_scale is not None, "use_fp4_cache=True requires q_scale"
372:     else:
373:         assert q_scale is None, "q_scale must be None when use_fp4_cache=False"
374: 
375:     # During speculative decoding, k may be padded to the CUDA graph batch
376:     # size while slot_mapping only covers actual tokens. Truncate k to avoid
377:     # out-of-bounds reads in the kernel.
378:     num_tokens = slot_mapping.shape[0]
379:     if k is not None:
380:         k = k[:num_tokens]
381: 
382:     if not skip_k_cache_insert:
383:         # scale_fmt can be None, but the function expects str
384:         assert scale_fmt is not None
385:         assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
386:         ops.indexer_k_quant_and_cache(
387:             k,
388:             kv_cache,
389:             slot_mapping,
390:             quant_block_size,
391:             scale_fmt,
392:         )
393: 
394:     # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
395:     # top-k kernels scatter valid indices into it. On the fused deepseek_v32
396:     # nvidia path, _fused_norm_rope_kernel already cleared the same
397:     # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
398:     # fill.
399:     if not skip_topk_buffer_clear:
400:         topk_indices_buffer[: hidden_states.shape[0]] = -1
```

### vllm/model_executor/layers/sparse_attn_indexer.py:495-650
```
495:                 row_starts=chunk.cu_seqlen_ks,
496:             )
497: 
498:     if has_decode:
499:         decode_metadata = attn_metadata_narrowed.decode
500:         assert decode_metadata is not None
501:         kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
502:         decode_lens = decode_metadata.decode_lens
503:         if decode_metadata.requires_padding:
504:             # pad in edge case where we have short chunked prefill length <
505:             # decode_threshold since we unstrictly split
506:             # prefill and decode by decode_threshold
507:             # (currently set to 1 + speculative tokens).
508:             # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
509:             # downstream context_lens masks stale slots). MXFP4 Q is two
510:             # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
511:             # packer with pad_byte=0 so padded slots dequantize to 0 and
512:             # can't produce NaN/Inf in the logits kernel.
513:             if q_scale is not None:
514:                 padded_q_quant_decode_tokens = pack_seq_triton(
515:                     q_quant[:num_decode_tokens], decode_lens, pad_value=0
516:                 )
517:                 padded_q_scale = pack_seq_triton(
518:                     q_scale[:num_decode_tokens], decode_lens, pad_value=0
519:                 )
520:             else:
521:                 padded_q_quant_decode_tokens = pack_seq_triton(
522:                     q_quant[:num_decode_tokens], decode_lens
523:                 )
524:                 padded_q_scale = None
525:         else:
526:             padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
527:                 decode_lens.shape[0], -1, *q_quant.shape[1:]
528:             )
529:             if q_scale is not None:
530:                 padded_q_scale = q_scale[:num_decode_tokens].reshape(
531:                     decode_lens.shape[0], -1, *q_scale.shape[1:]
532:                 )
533:             else:
534:                 padded_q_scale = None
535:         # TODO: move and optimize below logic with triton kernels
536:         batch_size = padded_q_quant_decode_tokens.shape[0]
537:         next_n = padded_q_quant_decode_tokens.shape[1]
538:         num_padded_tokens = batch_size * next_n
539:         seq_lens = decode_metadata.seq_lens[:batch_size]
540:         # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
541:         # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
542:         # the downstream topk kernels accept both 1D and 2D.
543:         padded_q_quant_cast = (
544:             padded_q_quant_decode_tokens.view(torch.int8)
545:             if use_fp4_cache
546:             else padded_q_quant_decode_tokens
547:         )
548:         if current_platform.is_xpu():
549:             if padded_q_scale is not None:
550:                 raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
551:             seq_lens_xpu = (
552:                 seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
553:             )
554:             logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
555:                 padded_q_quant_cast,
556:                 kv_cache,
557:                 weights[:num_padded_tokens],
558:                 seq_lens_xpu,
559:                 decode_metadata.block_table,
560:                 decode_metadata.schedule_metadata,
561:                 max_model_len,
562:             )
563:         else:
564:             logits = fp8_fp4_paged_mqa_logits(
565:                 (padded_q_quant_cast, padded_q_scale),
566:                 kv_cache,
567:                 weights[:num_padded_tokens],
568:                 seq_lens,
569:                 decode_metadata.block_table,
570:                 decode_metadata.schedule_metadata,
571:                 max_model_len=max_model_len,
572:                 clean_logits=False,
573:             )
574:         num_rows = logits.shape[0]
575:         topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
576: 
577:         use_cooperative_topk = (
578:             current_platform.is_cuda()
579:             and topk_tokens in (512, 1024, 2048)
580:             and num_rows <= 32
581:             and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
582:             and current_platform.has_device_capability(90)
583:             and not current_platform.is_device_capability_family(120)
584:         )
585:         use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
586:             512,
587:             1024,
588:             2048,
589:         )
590:         if use_cooperative_topk:
591:             workspace_manager = current_workspace_manager()
592:             (topk_workspace,) = workspace_manager.get_simultaneous(
593:                 ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
594:             )
595:             torch.ops._C.cooperative_topk(
596:                 logits,
597:                 seq_lens,
598:                 topk_indices,
599:                 topk_workspace,
600:                 topk_tokens,
601:                 attn_metadata_narrowed.max_seq_len,
602:             )
603:         elif use_persistent_topk:
604:             workspace_manager = current_workspace_manager()
605:             (topk_workspace,) = workspace_manager.get_simultaneous(
606:                 ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
607:             )
608:             torch.ops._C.persistent_topk(
609:                 logits,
610:                 seq_lens,
611:                 topk_indices,
612:                 topk_workspace,
613:                 topk_tokens,
614:                 logits.shape[1],
615:             )
616:         else:
617:             ops.top_k_per_row_decode(
618:                 logits,
619:                 next_n,
620:                 seq_lens,
621:                 topk_indices,
622:                 num_rows,
623:                 logits.stride(0),
624:                 logits.stride(1),
625:                 topk_tokens,
626:             )
627: 
628:         if decode_metadata.global_seq_lens is not None:
629:             _merge_dcp_topk_global(
630:                 logits,
631:                 topk_indices,
632:                 topk_tokens,
633:                 dcp_rank,
634:                 dcp_world_size,
635:                 cp_kv_cache_interleave_size,
636:             )
637: 
638:         if decode_metadata.requires_padding:
639:             # if padded, we need to unpack
640:             # the topk indices removing padded tokens
641:             topk_indices = unpack_seq_triton(
642:                 topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
643:                 decode_lens,
644:             )
645:             topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
646:                 topk_indices
647:             )
648: 
649:     return topk_indices_buffer
650: 
```

### vllm/model_executor/layers/sparse_attn_indexer.py:74-122
```
74: def _merge_dcp_topk_global(
75:     logits: torch.Tensor,
76:     topk_indices: torch.Tensor,
77:     topk_tokens: int,
78:     dcp_rank: int,
79:     dcp_world_size: int,
80:     cp_interleave: int,
81:     row_starts: torch.Tensor | None = None,
82: ) -> None:
83:     """Merge each DCP rank's local top-K into the global top-K.
84: 
85:     ``topk_indices`` are this rank's local top-K positions into its 1/N KV
86:     shard. A token in the global top-K must also be in its owning rank's local
87:     top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
88:     most that many on its own rank), so exchanging only the per-rank local
89:     candidates is exact -- equivalent to all-gathering the full logit matrix,
90:     but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
91:     score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
92:     padding); the attention backend localizes them back to physical slots per
93:     rank.
94:     """
95:     if dcp_world_size <= 1:
96:         return
97: 
98:     # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
99:     # (score, global_id) candidates on-device, all-gather, then the CuteDSL
100:     # stable-topk selector.
101:     _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
102:     from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
103:         pack_dcp_topk_candidates_cutedsl,
104:         stable_topk_from_gathered_candidates_cutedsl,
105:     )
106: 
107:     packed = torch.empty(
108:         (*topk_indices.shape, 2),
109:         dtype=torch.float32,
110:         device=topk_indices.device,
111:     )
112:     pack_dcp_topk_candidates_cutedsl(
113:         logits,
114:         topk_indices,
115:         packed,
116:         dcp_rank,
117:         dcp_world_size,
118:         cp_interleave,
119:         row_starts,
120:     )
121:     gathered = get_dcp_group().all_gather(packed, dim=1)
122:     indexer_fork.gathered()
```

### vllm/model_executor/layers/sparse_attn_indexer.py:655-690
```
655:     kv_cache: torch.Tensor,
656:     q_quant: torch.Tensor,
657:     q_scale: torch.Tensor | None,
658:     k: torch.Tensor,
659:     weights: torch.Tensor,
660:     quant_block_size: int,
661:     scale_fmt: str | None,
662:     topk_tokens: int,
663:     head_dim: int,
664:     max_model_len: int,
665:     total_seq_lens: int,
666:     topk_indices_buffer: torch.Tensor | None,
667:     skip_k_cache_insert: bool,
668:     use_fp4_cache: bool = False,
669:     dcp_rank: int = 0,
670:     dcp_world_size: int = 1,
671:     cp_kv_cache_interleave_size: int = 1,
672:     skip_topk_buffer_clear: bool = False,
673: ) -> torch.Tensor:
674:     return topk_indices_buffer
675: 
676: 
677: direct_register_custom_op(
678:     op_name="sparse_attn_indexer",
679:     op_func=sparse_attn_indexer,
680:     mutates_args=["topk_indices_buffer"],
681:     fake_impl=sparse_attn_indexer_fake,
682:     dispatch_key=current_platform.dispatch_key,
683: )
684: 
685: 
686: @CustomOp.register("sparse_attn_indexer")
687: class SparseAttnIndexer(CustomOp):
688:     """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
689:     separate custom op since it involves heavy custom kernels like `mqa_logits`,
690:     `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
```

### vllm/distributed/device_communicators/one_shot/__init__.py:82-178
```
82:     def _usable(self, inp: torch.Tensor, block_bytes: int) -> bool:
83:         nbytes = inp.numel() * inp.element_size()
84:         return (
85:             not self.ca.disabled
86:             and inp.is_contiguous()
87:             and block_bytes % 16 == 0
88:             and nbytes * self.world_size < self.ca.max_size
89:         )
90: 
91:     def _run(self, op, inp: torch.Tensor, out: torch.Tensor, *shape: int) -> None:
92:         if self.ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
93:             # captured: the input's own address is registered after capture
94:             op(self.ca._ptr, inp, out, *shape, 0, 0)
95:         elif self.ca._IS_CAPTURING:
96:             return  # warm-up inside a capture context: only the allocation matters
97:         else:
98:             op(
99:                 self.ca._ptr,
100:                 inp,
101:                 out,
102:                 *shape,
103:                 self.ca.buffer_ptrs[self.ca.rank],
104:                 self.ca.max_size,
105:             )
106: 
107:     def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor | None:
108:         """Concatenate every rank's `inp` along `dim`, or None if unsuitable."""
109:         dim %= inp.dim()
110:         outer = inp.shape[:dim].numel()
111:         block = inp.shape[dim:].numel() * inp.element_size()
112:         if not self._usable(inp, block):
113:             return None
114:         shape = list(inp.shape)
115:         shape[dim] *= self.world_size
116:         out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
117:         self._run(_extension().all_gather, inp, out, outer, block)
118:         return out
119: 
120:     def reduce_scatter(self, inp: torch.Tensor, dim: int) -> torch.Tensor | None:
121:         """Sum over ranks, keeping this rank's slice of `dim`; None if unsuitable."""
122:         dim %= inp.dim()
123:         if inp.shape[dim] % self.world_size:
124:             return None
125:         outer = inp.shape[:dim].numel()
126:         inner = inp.shape[dim:].numel() // self.world_size
127:         if inp.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not (
128:             self._usable(inp, inner * inp.element_size())
129:         ):
130:             return None
131:         shape = list(inp.shape)
132:         shape[dim] //= self.world_size
133:         out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
134:         self._run(_extension().reduce_scatter, inp, out, outer, inner)
135:         return out
136: 
137:     def _run2(self, op, a: torch.Tensor, b: torch.Tensor, *args) -> None:
138:         if self.ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
139:             op(self.ca._ptr, a, b, *args, 0, 0)
140:         elif not self.ca._IS_CAPTURING:
141:             op(
142:                 self.ca._ptr,
143:                 a.contiguous(),
144:                 b.contiguous(),
145:                 *args,
146:                 self.ca.buffer_ptrs[self.ca.rank],
147:                 self.ca.max_size,
148:             )
149: 
150:     def all_gather_cat(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
151:         """`all_gather(torch.cat([a, b], -1), dim=1)` for [T, H, *] inputs, in
152:         one kernel; None if unsuitable."""
153:         if a.dim() != 3 or b.shape[:2] != a.shape[:2] or a.dtype != b.dtype:
154:             return None
155:         es = a.element_size()
156:         if (
157:             a.stride(2) != 1
158:             or b.stride(2) != 1
159:             or any(
160:                 (v * es) % 16
161:                 for v in (a.shape[2], b.shape[2], *a.stride()[:2], *b.stride()[:2])
162:             )
163:         ):
164:             return None
165:         if a.data_ptr() % 16 or b.data_ptr() % 16:
166:             return None
167:         nbytes = (a.numel() + b.numel()) * es
168:         if self.ca.disabled or nbytes * self.world_size + 256 >= self.ca.max_size:
169:             return None
170:         T, H = a.shape[:2]
171:         out = torch.empty(
172:             (T, H * self.world_size, a.shape[2] + b.shape[2]),
173:             dtype=a.dtype,
174:             device=a.device,
175:         )
176:         self._run2(_extension().all_gather_cat, a, b, out)
177:         return out
178: 
```

### vllm/v1/worker/workspace.py:85-150
```
85:                 ],
86:             )
87: 
88:     def is_locked(self) -> bool:
89:         """Check if workspace is locked."""
90:         return self._locked
91: 
92:     def get_simultaneous(
93:         self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
94:     ) -> list[torch.Tensor]:
95:         """Get multiple workspace tensors simultaneously from a single allocation.
96: 
97:         Args:
98:             *shapes_and_dtypes: One or more (shape, dtype) tuples.
99: 
100:         Returns:
101:             List of tensor views into the workspace buffer, one per shape/dtype pair.
102:         """
103:         actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
104:         aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
105:         total_bytes = sum(aligned_bytes)
106: 
107:         # Calculate cumulative offsets using itertools.accumulate
108:         offsets = list(accumulate([0] + aligned_bytes[:-1]))
109: 
110:         current_workspace = self._ensure_workspace_size(total_bytes)
111: 
112:         return [
113:             current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
114:             .view(shapes_and_dtypes[i][1])
115:             .reshape(shapes_and_dtypes[i][0])
116:             for i in range(len(shapes_and_dtypes))
117:         ]
118: 
119:     def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
120:         """Ensure workspace is allocated and large enough, return current workspace.
121: 
122:         Args:
123:             required_bytes: The number of bytes required.
124: 
125:         Returns:
126:             The current workspace tensor.
127:         """
128:         ubatch_id = dbo_current_ubatch_id()
129:         current_workspace = self._current_workspaces[ubatch_id]
130:         current_size = self._workspace_size_bytes(current_workspace)
131: 
132:         if current_size < required_bytes:
133: 
134:             def get_caller_info() -> str:
135:                 """Find first frame outside WorkspaceManager."""
136:                 curr_frame = inspect.currentframe()
137:                 if curr_frame is None:
138:                     return "unknown"
139:                 # Walk up the stack skipping WorkspaceManager frames
140:                 curr_frame = curr_frame.f_back
141:                 while curr_frame is not None:
142:                     # TODO: This only catches instance methods (self), missing
143:                     # classmethods and staticmethods. Once Python 3.11+ is the
144:                     # minimum supported version, use co_qualname instead:
145:                     #   qualname = curr_frame.f_code.co_qualname
146:                     #   if qualname.startswith("WorkspaceManager."):
147:                     if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
148:                         curr_frame = curr_frame.f_back
149:                         continue
150:                     filename = os.path.basename(curr_frame.f_code.co_filename)
```

### csrc/custom_all_reduce.cuh:195-245
```
195: // This function is meant to be used as the first synchronization in the all
196: // reduce kernel. Thus, it doesn't need to make any visibility guarantees for
197: // prior memory accesses. Note: volatile writes will not be reordered against
198: // other volatile writes.
199: template <int ngpus>
200: DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
201:                               int rank) {
202:   uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
203:   if (threadIdx.x < ngpus) {
204:     auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
205:     auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
206:     // Write the expected counter value to peer and wait for correct value
207:     // from peer.
208:     st_flag_volatile(peer_counter_ptr, flag);
209:     while (ld_flag_volatile(self_counter_ptr) != flag);
210:   }
211:   __syncthreads();
212:   // use one thread to update flag
213:   if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
214: }
215: 
216: // This function is meant to be used as the second or the final
217: // synchronization barrier in the all reduce kernel. If it's the final
218: // synchronization barrier, we don't need to make any visibility guarantees
219: // for prior memory accesses.
220: template <int ngpus, bool final_sync = false>
221: DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
222:   __syncthreads();
223:   uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
224:   if (threadIdx.x < ngpus) {
225:     auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
226:     auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
227:     // Write the expected counter value to peer and wait for correct value from
228:     // peer.
229:     if constexpr (!final_sync) {
230:       st_flag_release(peer_counter_ptr, flag);
231:       while (ld_flag_acquire(self_counter_ptr) != flag);
232:     } else {
233:       st_flag_volatile(peer_counter_ptr, flag);
234:       while (ld_flag_volatile(self_counter_ptr) != flag);
235:     }
236:   }
237:   if constexpr (!final_sync) __syncthreads();
238: 
239:   // use one thread to update flag
240:   if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
241: }
242: 
243: #else
244: 
245: template <int ngpus>
```

### vllm/v1/attention/backends/mla/flashmla_sparse.py:880-995
```
880:     def _forward_fp8_kv_mixed_batch(
881:         self,
882:         q: torch.Tensor,
883:         kv_c_and_k_pe_cache: torch.Tensor,
884:         topk_indices: torch.Tensor,
885:         attn_metadata: FlashMLASparseMetadata,
886:     ) -> tuple[torch.Tensor, torch.Tensor]:
887:         """Mixed batch FP8 forward path that treats all tokens as one batch.
888: 
889:         This is equivalent to main branch's approach and avoids the BF16
890:         prefill kernel which has head padding overhead when num_heads is small.
891:         Used when use_mixed_batch is True.
892:         """
893:         # Convert per-request indices to global slots (decode) or workspace
894:         # offsets (prefill). Under DCP, keep only this rank's KV shard: owned
895:         # slots are compacted to a contiguous prefix and the tail stays -1,
896:         # which the kernel skips; the LSE-based cross-rank combine restores
897:         # the full softmax.
898:         if (
899:             attn_metadata.dcp_prefill_local_len > 0
900:             and q.shape[0] == attn_metadata.num_actual_tokens
901:             and not torch.cuda.is_current_stream_capturing()
902:         ):
903:             return self._forward_dcp_sparse_prefill(
904:                 q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
905:             )
906:         grace_cache = kv_c_and_k_pe_cache
907:         hot_rows: torch.Tensor | None = None
908:         pending, self._hot_pending = self._hot_pending, None
909:         if pending is not None:
910:             topk_indices, valid_counts, sched, hot_rows, ready = pending
911:             assert (sched is not None) == (
912:                 self.dcp_world_size > 1 and self._dcp_decode_sched(q)
913:             )
914:             torch.cuda.current_stream().wait_event(ready)
915:         else:
916:             topk_indices, valid_counts, sched = self._converted_indices(
917:                 topk_indices, attn_metadata, self._dcp_decode_sched(q)
918:             )
919:             hot = self._hot_cache(kv_c_and_k_pe_cache, attn_metadata)
920:             if hot is not None:
921:                 hot_rows = hot.map(
922:                     topk_indices, attn_metadata.slot_mapping[: q.shape[0]]
923:                 )
924:         if hot_rows is not None:
925:             hot = anchor_hot_cache.get(grace_cache)
926:             assert hot is not None
927:             kv_c_and_k_pe_cache = hot.buffer
928:         stager = (
929:             skip_kv_stage.get_stager(q.device)
930:             if self._skip_kv
931:             and attn_metadata.num_prefills == 0
932:             and q.shape[0] <= skip_kv_stage.limits()[0]
933:             else None
934:         )
935:         if (
936:             stager is not None
937:             and self._reuses_topk
938:             and (staged := stager.staged(kv_c_and_k_pe_cache, q.shape[0]))
939:         ):
940:             kv_c_and_k_pe_cache, topk_indices = staged
941: 
942:         assert attn_metadata.fp8_extra_metadata is not None
943:         assert isinstance(
944:             attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
945:         )
946:         fp8_metadata = attn_metadata.fp8_extra_metadata
947: 
948:         attn_indices = topk_indices if hot_rows is None else hot_rows
949:         if sched is not None:
950:             _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
951:                 q.unsqueeze(0),
952:                 kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
953:                 attn_indices.unsqueeze(0),
954:                 None,
955:                 None,
956:                 *sched,
957:                 None,
958:                 None,
959:                 None,
960:                 512,
961:                 self.softmax_scale,
962:                 None,
963:             )
964:         else:
965:             _attn_out, _lse = self._fp8_flash_mla_kernel(
966:                 q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
967:                 kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
968:                 topk_indices=attn_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
969:                 kernel_metadata=fp8_metadata,
970:             )
971: 
972:         if hot_rows is not None and (check := anchor_hot_cache.check()) is not None:
973:             assert sched is not None
974:             ref_out, ref_lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
975:                 q.unsqueeze(0),
976:                 grace_cache.view(torch.uint8).unsqueeze(-2),
977:                 topk_indices.unsqueeze(0),
978:                 None,
979:                 None,
980:                 *sched,
981:                 None,
982:                 None,
983:                 None,
984:                 512,
985:                 self.softmax_scale,
986:                 None,
987:             )
988:             check(_attn_out, ref_out, _lse, ref_lse)
989: 
990:         if stager is not None and not self._reuses_topk:
991:             # after the anchor's own attention, so the copy does not slow it
992:             stager.stage(grace_cache, topk_indices, attn_metadata.slot_mapping)
993: 
994:         # Output is (1, T, H, D_v) -> (T, H, D_v); lse is (1, H, T) -> (T, H)
995:         lse = _lse.squeeze(0).transpose(0, 1)
```

### vllm/v1/attention/backends/mla/skip_kv_stage.py:231-300
```
231:     def stage(
232:         self, anchor_cache: torch.Tensor, indices: torch.Tensor, new_slots: torch.Tensor
233:     ) -> None:
234:         """Anchor, after its FlashMLA launch: start its group's copy."""
235:         entry = self.groups.get(anchor_cache.data_ptr())
236:         if entry is None:
237:             return
238:         group, ready, number = entry
239:         t, n = indices.shape[0], indices.numel()
240:         # every entry may be a distinct row: the buffers must hold them all
241:         assert t <= self.max_queries and indices.shape[1] <= self.columns, (
242:             f"skip-KV staging sized for {self.max_queries} x {self.columns}, "
243:             f"got {tuple(indices.shape)}"
244:         )
245:         self._width = indices.shape[1]
246:         new = new_slots[:t]
247:         if number == 0:  # this step's tokens are the same for every group
248:             self.reserved_map[:t] = torch.where(new >= 0, self._reserved[:t], -1)
249:         self.stream.wait_stream(torch.cuda.current_stream())
250:         with torch.cuda.stream(self.stream):
251:             self.count.zero_()
252:             grid = (triton.cdiv(n, 1024),)
253:             _mark_rows_kernel[grid](indices, n, new, self.row_of, T=t, BLOCK=1024)
254:             num_slots = self.row_of.numel() - 1
255:             _compact_rows_kernel[(triton.cdiv(num_slots, 4096),)](
256:                 self.row_of, num_slots, self.slots, self.count, CHUNK=4096
257:             )
258:             _remap_rows_kernel[grid](
259:                 indices,
260:                 n,
261:                 new,
262:                 self.row_of,
263:                 self.rows,
264:                 T=t,
265:                 RESERVED=self.gather_rows,
266:                 BLOCK=1024,
267:             )
268:             srcs = [c.view(torch.int32) for c in group]
269:             srcs += [srcs[0]] * (STAGE_BUFFERS - len(srcs))
270:             words = ROW_BYTES // 4
271:             _gather_rows_kernel[(GATHER_CTAS, len(group))](
272:                 *srcs,
273:                 self.buffers.view(torch.int32),
274:                 self.slots,
275:                 self.count,
276:                 self.row_of,
277:                 DST_WORDS=(self.gather_rows + RESERVED_ROWS) * words,
278:                 ROW_WORDS=words,
279:                 BLOCK_R=32,
280:                 BLOCK_W=triton.next_power_of_2(words),
281:             )
282:             if self.hist is not None:
283:                 self.hist[number].index_add_(
284:                     0, (self.count // 256).long(), self.count.new_ones(1).float()
285:                 )
286:             ready.record(self.stream)
287:         indices.record_stream(self.stream)
288:         new.record_stream(self.stream)
289:         self._ready = ready
290: 
291:     def staged(
292:         self, cache: torch.Tensor, num_queries: int
293:     ) -> tuple[torch.Tensor, torch.Tensor] | None:
294:         """Skip layer: its staged buffer and remapped indices, else None."""
295:         p = self.position.get(cache.data_ptr())
296:         if p is None:
297:             return None
298:         assert self._ready is not None
299:         torch.cuda.current_stream().wait_event(self._ready)
300:         rows = self.rows.view(-1)[: num_queries * self._width]
```

### vllm/model_executor/layers/mla.py:157-330
```
157:     def forward(
158:         self,
159:         positions: torch.Tensor,
160:         hidden_states: torch.Tensor,
161:         llama_4_scaling: torch.Tensor | None = None,
162:     ) -> torch.Tensor:
163:         q_c = None
164:         kv_lora = None
165:         if self.indexer_fork:
166:             torch.ops.vllm.mla_indexer_fork_k(
167:                 hidden_states,
168:                 self.topk_indices_buffer,
169:                 _encode_layer_name(self.mla_attn.layer_name),
170:             )
171: 
172:         if self.qkv_glue:
173:             assert self.fused_qkv_a_proj is not None
174:             assert self.q_a_layernorm is not None and self.q_b_proj is not None
175:             q_c, kv_c_normed, k_pe = torch.ops.vllm.mla_qkv_glue(
176:                 self.fused_qkv_a_proj(hidden_states)[0],
177:                 positions,
178:                 self.rotary_emb.cos_sin_cache,
179:                 self.q_a_layernorm.weight,
180:                 self.kv_a_layernorm.weight,
181:                 self.q_a_layernorm.variance_epsilon,
182:                 self.kv_a_layernorm.variance_epsilon,
183:                 _encode_layer_name(self.mla_attn.layer_name),
184:             )
185:             self._fork_indexer_q(hidden_states, q_c, positions)
186:             q = self.q_b_proj(q_c)[0].view(-1, self.num_heads, self.qk_head_dim)
187:             k_pe = k_pe.unsqueeze(1)
188:             q[..., self.qk_nope_head_dim :], _ = self.rotary_emb(
189:                 positions, q[..., self.qk_nope_head_dim :]
190:             )
191:         elif self.q_lora_rank is not None:
192:             assert self.fused_qkv_a_proj is not None, (
193:                 "fused_qkv_a_proj is required when q_lora_rank is not None"
194:             )
195:             assert self.q_a_layernorm is not None, (
196:                 "q_a_layernorm is required when q_lora_rank is not None"
197:             )
198:             assert self.q_b_proj is not None, (
199:                 "q_b_proj is required when q_lora_rank is not None"
200:             )
201: 
202:             qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
203:             q_c, kv_lora = qkv_lora.split(
204:                 [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
205:                 dim=-1,
206:             )
207:             q_c = self.q_a_layernorm(q_c)
208:             self._fork_indexer_q(hidden_states, q_c, positions)
209:             q = self.q_b_proj(q_c)[0]
210:         else:
211:             assert self.kv_a_proj_with_mqa is not None, (
212:                 "kv_a_proj_with_mqa is required when q_lora_rank is None"
213:             )
214:             assert self.q_proj is not None, (
215:                 "q_proj is required when q_lora_rank is None"
216:             )
217:             kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
218:             q = self.q_proj(hidden_states)[0]
219: 
220:         if not self.qkv_glue:
221:             assert kv_lora is not None
222:             kv_c, k_pe = kv_lora.split(
223:                 [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
224:             )
225:             kv_c_normed = self.kv_a_layernorm(kv_c)
226: 
227:             q = q.view(-1, self.num_heads, self.qk_head_dim)
228:             # Add head dim of 1 to k_pe
229:             k_pe = k_pe.unsqueeze(1)
230: 
231:             if self.rotary_emb is not None:
232:                 q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
233:                     positions, q[..., self.qk_nope_head_dim :], k_pe
234:                 )
235: 
236:         if self.cold_prefetch_layer is not None:
237:             torch.ops.vllm.tiered_moe_cold_join(q)
238: 
239:         if (
240:             self.indexer
241:             and self.is_sparse
242:             and not self.skip_topk
243:             and not self.indexer_fork
244:         ):
245:             self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
246: 
247:         if llama_4_scaling is not None:
248:             q *= llama_4_scaling
249: 
250:         attn_out = self.mla_attn(
251:             q,
252:             kv_c_normed,
253:             k_pe,
254:             output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
255:         )
256: 
257:         if self.cold_prefetch_layer is not None:
258:             torch.ops.vllm.tiered_moe_cold_fork(attn_out, self.cold_prefetch_layer)
259:         return self.o_proj(attn_out)[0]
```
