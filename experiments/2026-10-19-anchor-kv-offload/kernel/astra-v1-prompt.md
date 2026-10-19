You are reviewing a CUDA kernel design for vLLM. Do NOT run any commands (the sandbox cannot start here); everything you need is inlined below. Answer the questions at the end of the brief, most severe issues first, with concrete fixes.

# Anchor hot cache: review brief

## Context

GLM-5.3 (DSA sparse attention, MLA, fp8_ds_mla KV 656 B/token/layer) served
on 4x GH200 with vLLM, TP4 + DCP4 (KV tokens sharded over 4 ranks), MTP3
speculative decode (4 query tokens per request per step), c=4 requests,
CUDA graphs (FULL graphs for decode). Tiered MoE: experts that don't fit in
HBM live on Grace; every GiB of HBM freed = ~200 more hot experts.

KV tier today (`mla_cache_tier=skip_host_uva`): the 21 target "anchor"
layers (own DSA indexer) + MTP layer 78 keep their main KV in HBM; the 57
"skip" layers (reuse the previous anchor's top-k) keep theirs in pinned Grace
memory (UVA). After each anchor's FlashMLA, a side stream (`SkipKVStager`,
skip_kv_stage.py) copies the rows the anchor's top-k selects for its 3 skip
layers into HBM staging buffers and remaps the indices; skip layers write
this step's tokens into reserved rows. Anchor main KV at the 800K pool:
2.69 GiB per rank.

Goal: move the 21 anchors' main KV to Grace too, with an HBM hot buffer per
anchor, at ~zero decode cost. Same ops only: FlashMLA must read the same
bytes in the same order (indices keep position, multiplicity and -1), so the
attention output is bitwise identical.

## Evidence so far

- Top-k dumps from real c=4 agentic traffic (17K decode steps x 2 runs,
  contexts ~290-330K global, 73% of steps M=16 tokens): per anchor and step
  the batch reads ~3558 distinct rows per rank (M=16), ~839 (M=4); each
  token selects ~519 valid of 768 columns (DCP-converted, rank-local).
- LRU sim: one buffer per layer shared by the batch, 16384 rows: 6.2% miss
  (~220 rows/anchor call, p99 ~780); per-request buffers 4096 rows each:
  7.4%. CLOCK (ref bit, skip rows read this step) matches exact LRU; FIFO
  is ~1-1.5 pp worse. Inserting this step's new tokens into real rows (vs
  reserved rows) changes nothing.
- HBM: 21 x 16384 x 656 B = 215 MiB vs 2.69 GiB: ~2.48 GiB/rank freed.
- Critical-path window: from the end of the DCP top-k merge kernel to the
  anchor's FlashMLA start there is 18.8 us (M=4) / 20.5 us (M=16), of which
  10.8 / 14.8 us is top-k-independent work (KV write, W_UK absorb GEMM, q
  concat) and ~5 us the index conversion + schedule. A side stream forked at
  the merge could hide ~11-15 us. FlashMLA itself is ~10-13 us.
- Today's stager kernels for comparison (mark 2-3.6, compact 5, remap 2-4.6,
  gather of 3 layers 15-59 us).

## Proposed design (v1)

Per anchor layer, on HBM: buffer H (N=16384 rows + 64 reserved rows, laid out
as the FlashMLA cache, (rows/64, 64, 656)), `where[S]` (slot -> row; S ~ 200K
slots per rank), `slot_of[N]`, `stamp[N]` (step last read), `ref[N]` (CLOCK
bit), counters. Global per layer (not per request): prefix-shared blocks stay
cached across turns.

Per anchor per decode step, after the DCP index conversion and before
FlashMLA (anchor_hot.cu, 4 launches):
1. lookup: invalidate this step's new-token slots (KV write changes them);
   entries: -1 -> -1, new token j -> N + j (reserved row, filled by a second
   KV write into H, as the skip layers do), hit -> row (stamp, ref=1), miss ->
   pending list; first miss per slot claims where[s] -1 -> -2 (atomicCAS) and
   is listed. Warp-aggregated appends.
2. select: per warp 32 claims; CLOCK candidates from one atomic per round on
   the hand; skip rows stamped this step; second chance on ref; victim owned
   via atomicExch(stamp, step); evict (where[old] = -1), publish where[s].
3. copy: Grace -> H for every missed row, one 16 B word per thread (41 per
   row), 4 loads before 4 stores; then pending entries read where[s].
4. finish: step++, counters reset.

Planned integration:
- New tier (anchors' main caches on Grace; layer 78 stays in HBM).
- KV write: normal write goes to Grace (UVA); a second write puts this
  step's tokens into H's reserved rows (top-k independent, in the window).
- Invalidation outside decode: prefill / any KV write to slot s must drop
  where[s] in all 21 layers (blocks freed and reallocated to another request
  would otherwise be served stale). Plan: a scatter kernel over the step's
  slot_mapping on every step that writes KV (prefill and decode).
- Prefill reads the anchor cache from Grace through the existing full-shard
  upconvert gather (costs ~63 us per anchor per 56K-token chunk, ~0.2%).
- Placement: fork lookup/select/copy on a side stream right after the DCP
  top-k merge (needs the index conversion moved there too), join before
  FlashMLA.
- Skip layers unchanged (they stage from their own Grace caches using the
  anchor's converted indices, which stay slot-valued).

## Questions for review

1. Correctness of anchor_hot.cu: races between lookup/select/copy, the claim
   protocol, victim ownership (hand wrap within a step), eviction of a slot
   that another entry hit this step, the -2 state on overflow, memory
   ordering between kernels, int overflow of the hand.
2. Design holes: speculative decode (rejected draft slots rewritten next
   step), block free/realloc, prefix-cache hits on blocks written by another
   request, DCP (slot_mapping is -1 for tokens other ranks own), padded FULL
   graph batches, CUDA graph capture of the counters, the MTP drafter reading
   anchor layers?, prefill/decode mixing.
3. Cost: is 4 launches + atomics realistic inside ~10-15 us at ~220 (p99
   ~800, max ~3500) misses per call? What would you fuse or restructure
   (cooperative single kernel, persistent kernel, PDL)? Better policy than
   CLOCK for a GPU?
4. Anything simpler that reaches the same HBM saving.

## anchor_hot.cu (v1, numbered)
```cuda
     1	// Anchor hot cache: an HBM buffer of N rows in front of an anchor layer's
     2	// fp8_ds_mla main cache on Grace. Per decode step and anchor, the converted
     3	// top-k (rank-local slots) becomes buffer rows:
     4	//   lookup  hits stamp their row and set its CLOCK reference bit; misses claim
     5	//           their slot (where[s]: -1 -> -2, one winner per slot) and are
     6	//           listed; this step's tokens map to reserved rows N + j and their
     7	//           slots are invalidated (the KV write changes the slot's content)
     8	//   select  each claimed slot takes a CLOCK victim not read this step and
     9	//           publishes where[s]
    10	//   copy    missed rows Grace -> HBM, every 16 B word its own thread; entries
    11	//           that missed read their row
    12	//   finish  next step
    13	// Entries keep position, multiplicity and -1, so FlashMLA reads the same
    14	// bytes in the same order as from the full cache.
    15	#include <cuda_runtime.h>
    16	#include <stdint.h>
    17	#include <torch/extension.h>
    18	#include <c10/cuda/CUDAStream.h>
    19	
    20	namespace {
    21	
    22	constexpr int ROW_WORDS = 656 / 16;  // 41 x 16 B
    23	
    24	// ctl: [0] step, [1] CLOCK hand, [2] claimed slots, [3] pending entries,
    25	// [4] overflow (a claim without a victim: must stay 0), [5] candidates swept
    26	struct State {
    27	  int32_t* where;     // [S] slot -> row, -1 absent, -2 claimed this step
    28	  int32_t* slot_of;   // [N] row -> slot, -1 free
    29	  int32_t* stamp;     // [N] step the row was last read or filled
    30	  int8_t* ref;        // [N] CLOCK reference bit
    31	  int32_t* ctl;
    32	  int32_t* miss;      // [MAXE] claimed slots
    33	  int32_t* miss_row;  // [MAXE] their rows
    34	  int32_t* pending;   // [MAXE] entries waiting for their row
    35	};
    36	
    37	__device__ __forceinline__ int warp_append(int32_t* counter, bool want) {
    38	  unsigned m = __ballot_sync(0xffffffffu, want);
    39	  if (!m) return -1;
    40	  int lane = threadIdx.x & 31;
    41	  int leader = __ffs(m) - 1;
    42	  int base = 0;
    43	  if (lane == leader) base = atomicAdd(counter, __popc(m));
    44	  base = __shfl_sync(0xffffffffu, base, leader);
    45	  return want ? base + __popc(m & ((1u << lane) - 1)) : -1;
    46	}
    47	
    48	__global__ void lookup_kernel(State st, const int32_t* __restrict__ idx, int n,
    49	                              const int32_t* __restrict__ new_slots, int t,
    50	                              int32_t* __restrict__ out, int nrows) {
    51	  __shared__ int32_t sh_new[64];
    52	  if (threadIdx.x < 64) sh_new[threadIdx.x] = threadIdx.x < t ? new_slots[threadIdx.x] : -1;
    53	  __syncthreads();
    54	  const int step = st.ctl[0] + 1;
    55	  if (blockIdx.x == 0 && threadIdx.x < t) {
    56	    // no entry reads where[] for these slots (they map to reserved rows)
    57	    int s = sh_new[threadIdx.x];
    58	    if (s >= 0) {
    59	      int r = st.where[s];
    60	      if (r >= 0) {
    61	        st.where[s] = -1;
    62	        st.slot_of[r] = -1;
    63	        st.ref[r] = 0;
    64	      }
    65	    }
    66	  }
    67	  for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
    68	    int e = base + threadIdx.x;
    69	    int s = e < n ? idx[e] : -1;
    70	    int o = -1;
    71	    bool pend = false, claim = false;
    72	    if (s >= 0) {
    73	      int j = -1;
    74	      for (int k = 0; k < t; ++k) j = (sh_new[k] == s) ? k : j;
    75	      if (j >= 0) {
    76	        o = nrows + j;
    77	      } else {
    78	        int r = st.where[s];
    79	        if (r >= 0) {
    80	          o = r;
    81	          st.stamp[r] = step;
    82	          st.ref[r] = 1;
    83	        } else {
    84	          pend = true;
    85	          if (r == -1) claim = atomicCAS(&st.where[s], -1, -2) == -1;
    86	        }
    87	      }
    88	    }
    89	    if (e < n) out[e] = o;
    90	    int p = warp_append(&st.ctl[3], pend);
    91	    if (pend) st.pending[p] = e;
    92	    int m = warp_append(&st.ctl[2], claim);
    93	    if (claim) st.miss[m] = s;
    94	  }
    95	}
    96	
    97	// A warp serves up to 32 claims: per round it takes as many CLOCK candidates
    98	// as it still needs with one atomic on the hand; a candidate read this step is
    99	// skipped, one with its reference bit set gets a second chance. A victim is
   100	// owned through its stamp (atomicExch to this step), so a hand that wrapped
   101	// within the step can never give one row to two slots.
   102	__global__ void select_kernel(State st, int nrows) {
   103	  const int step = st.ctl[0] + 1;
   104	  const int nmiss = st.ctl[2];
   105	  const int lane = threadIdx.x & 31;
   106	  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
   107	  const int nwarps = (gridDim.x * blockDim.x) >> 5;
   108	  for (int b = warp * 32; b < nmiss; b += nwarps * 32) {
   109	    const int i = b + lane;
   110	    bool need = i < nmiss;
   111	    int victim = -1;
   112	    for (int round = 0; round < 64; ++round) {
   113	      unsigned needm = __ballot_sync(0xffffffffu, need);
   114	      if (!needm) break;
   115	      int k = __popc(needm);
   116	      unsigned c0 = 0;
   117	      if (lane == 0) {
   118	        c0 = atomicAdd(reinterpret_cast<unsigned*>(&st.ctl[1]), (unsigned)k);
   119	        atomicAdd(&st.ctl[5], k);
   120	      }
   121	      c0 = __shfl_sync(0xffffffffu, c0, 0);
   122	      int c = (int)((c0 + lane) % (unsigned)nrows);
   123	      bool ok = false;
   124	      if (lane < k) {
   125	        ok = st.stamp[c] != step;
   126	        if (ok && st.ref[c]) {
   127	          st.ref[c] = 0;
   128	          ok = false;
   129	        }
   130	        if (ok) ok = atomicExch(&st.stamp[c], step) != step;
   131	      }
   132	      unsigned okm = __ballot_sync(0xffffffffu, ok);
   133	      while (okm && needm) {
   134	        int from = __ffs(okm) - 1;
   135	        int to = __ffs(needm) - 1;
   136	        int cv = __shfl_sync(0xffffffffu, c, from);
   137	        if (lane == to) {
   138	          victim = cv;
   139	          need = false;
   140	        }
   141	        okm &= okm - 1;
   142	        needm &= needm - 1;
   143	      }
   144	    }
   145	    if (i < nmiss) {
   146	      int s = st.miss[i];
   147	      if (victim >= 0) {
   148	        int old = st.slot_of[victim];
   149	        if (old >= 0) st.where[old] = -1;
   150	        st.slot_of[victim] = s;
   151	        st.ref[victim] = 0;
   152	        st.where[s] = victim;
   153	      } else {
   154	        atomicAdd(&st.ctl[4], 1);
   155	        st.where[s] = -1;
   156	      }
   157	      st.miss_row[i] = victim;
   158	    }
   159	  }
   160	}
   161	
   162	template <int UNROLL>
   163	__global__ void copy_kernel(State st, const int4* __restrict__ src, int4* __restrict__ dst,
   164	                            const int32_t* __restrict__ idx, int32_t* __restrict__ out) {
   165	  const int nmiss = st.ctl[2];
   166	  const int np = st.ctl[3];
   167	  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
   168	  const int stride = gridDim.x * blockDim.x;
   169	  const int total = nmiss * ROW_WORDS;
   170	  for (int f0 = tid; f0 < total; f0 += stride * UNROLL) {
   171	    int4 v[UNROLL];
   172	    int64_t to[UNROLL];
   173	#pragma unroll
   174	    for (int u = 0; u < UNROLL; ++u) {
   175	      int f = f0 + u * stride;
   176	      to[u] = -1;
   177	      if (f < total) {
   178	        int i = f / ROW_WORDS, w = f - i * ROW_WORDS;
   179	        int r = st.miss_row[i];
   180	        if (r >= 0) {
   181	          v[u] = src[(int64_t)st.miss[i] * ROW_WORDS + w];
   182	          to[u] = (int64_t)r * ROW_WORDS + w;
   183	        }
   184	      }
   185	    }
   186	#pragma unroll
   187	    for (int u = 0; u < UNROLL; ++u)
   188	      if (to[u] >= 0) dst[to[u]] = v[u];
   189	  }
   190	  for (int p = tid; p < np; p += stride) {
   191	    int e = st.pending[p];
   192	    out[e] = st.where[idx[e]];
   193	  }
   194	}
   195	
   196	__global__ void finish_kernel(State st) {
   197	  st.ctl[0] += 1;
   198	  st.ctl[2] = 0;
   199	  st.ctl[3] = 0;
   200	}
   201	
   202	}  // namespace
   203	
   204	void anchor_hot_map(torch::Tensor idx, torch::Tensor new_slots, torch::Tensor out,
   205	                    torch::Tensor src, torch::Tensor dst, torch::Tensor where,
   206	                    torch::Tensor slot_of, torch::Tensor stamp, torch::Tensor ref,
   207	                    torch::Tensor ctl, torch::Tensor miss, torch::Tensor miss_row,
   208	                    torch::Tensor pending, int64_t select_ctas, int64_t copy_ctas) {
   209	  State st{where.data_ptr<int32_t>(), slot_of.data_ptr<int32_t>(),
   210	           stamp.data_ptr<int32_t>(), ref.data_ptr<int8_t>(),
   211	           ctl.data_ptr<int32_t>(),   miss.data_ptr<int32_t>(),
   212	           miss_row.data_ptr<int32_t>(), pending.data_ptr<int32_t>()};
   213	  auto stream = at::cuda::getCurrentCUDAStream();
   214	  const int n = idx.numel();
   215	  const int t = new_slots.numel();
   216	  TORCH_CHECK(t <= 64 && n <= miss.numel());
   217	  const int nrows = slot_of.numel();
   218	  const int lk = std::max(std::min((n + 255) / 256, 132), 1);
   219	  lookup_kernel<<<lk, 256, 0, stream>>>(st, idx.data_ptr<int32_t>(), n,
   220	                                         new_slots.data_ptr<int32_t>(), t,
   221	                                         out.data_ptr<int32_t>(), nrows);
   222	  select_kernel<<<select_ctas, 128, 0, stream>>>(st, nrows);
   223	  copy_kernel<4><<<copy_ctas, 256, 0, stream>>>(
   224	      st, reinterpret_cast<const int4*>(src.data_ptr()),
   225	      reinterpret_cast<int4*>(dst.data_ptr()), idx.data_ptr<int32_t>(),
   226	      out.data_ptr<int32_t>());
   227	  finish_kernel<<<1, 1, 0, stream>>>(st);
   228	}
   229	
   230	PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("anchor_hot_map", &anchor_hot_map); }
```

## vllm/v1/attention/backends/mla/flashmla_sparse.py:891-960 (decode path today)
```python
1+890	        ):
2+890	            return self._forward_dcp_sparse_prefill(
3+890	                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
4+890	            )
5+890	        valid_counts: torch.Tensor | None = None
6+890	        sched: tuple[torch.Tensor, torch.Tensor] | None = None
7+890	        if self.dcp_world_size > 1:
8+890	            topk_indices, valid_counts, sched = self._dcp_converted_indices(
9+890	                topk_indices, attn_metadata, decode_sched=self._dcp_decode_sched(q)
10+890	            )
11+890	        else:
12+890	            topk_indices = triton_convert_req_index_to_global_index(
13+890	                attn_metadata.req_id_per_token,
14+890	                attn_metadata.block_table,
15+890	                topk_indices,
16+890	                BLOCK_SIZE=attn_metadata.block_size,
17+890	                NUM_TOPK_TOKENS=topk_indices.shape[1],
18+890	            )
19+890	        stager = (
20+890	            skip_kv_stage.get_stager(q.device)
21+890	            if self._skip_kv
22+890	            and attn_metadata.num_prefills == 0
23+890	            and q.shape[0] <= skip_kv_stage.limits()[0]
24+890	            else None
25+890	        )
26+890	        if (
27+890	            stager is not None
28+890	            and self._reuses_topk
29+890	            and (staged := stager.staged(kv_c_and_k_pe_cache, q.shape[0]))
30+890	        ):
31+890	            kv_c_and_k_pe_cache, topk_indices = staged
32+890	
33+890	        assert attn_metadata.fp8_extra_metadata is not None
34+890	        assert isinstance(
35+890	            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
36+890	        )
37+890	        fp8_metadata = attn_metadata.fp8_extra_metadata
38+890	
39+890	        if sched is not None:
40+890	            _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
41+890	                q.unsqueeze(0),
42+890	                kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
43+890	                topk_indices.unsqueeze(0),
44+890	                None,
45+890	                None,
46+890	                *sched,
47+890	                None,
48+890	                None,
49+890	                None,
50+890	                512,
51+890	                self.softmax_scale,
52+890	                None,
53+890	            )
54+890	        else:
55+890	            _attn_out, _lse = self._fp8_flash_mla_kernel(
56+890	                q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
57+890	                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
58+890	                topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
59+890	                kernel_metadata=fp8_metadata,
60+890	            )
61+890	
62+890	        if stager is not None and not self._reuses_topk:
63+890	            # after the anchor's own attention, so the copy does not slow it
64+890	            stager.stage(kv_c_and_k_pe_cache, topk_indices, attn_metadata.slot_mapping)
65+890	
66+890	        # Output is (1, T, H, D_v) -> (T, H, D_v); lse is (1, H, T) -> (T, H)
67+890	        lse = _lse.squeeze(0).transpose(0, 1)
68+890	        if valid_counts is not None and not self.dcp_combine_drops_inf_lse:
69+890	            lse = mask_empty_dcp_lse(lse, valid_counts)
70+890	        return _attn_out.squeeze(0), lse
```

## vllm/v1/attention/backends/mla/skip_kv_stage.py (the existing skip-layer stager, full)
```python
     1	# SPDX-License-Identifier: Apache-2.0
     2	# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
     3	"""HBM staging for index-share layers whose main MLA cache lives on Grace.
     4	
     5	With ``--mla-cache-tier skip_host_uva`` only the anchor layers (those with their
     6	own DSA indexer) keep their fp8_ds_mla cache in HBM. The layers that reuse an
     7	anchor's top-k ("skip" layers) keep theirs in pinned Grace memory. In a decode
     8	step the anchor converts its top-k to this rank's physical slots; right after,
     9	a side stream copies the selected rows of each following skip layer into one
    10	shared set of HBM buffers and remaps the indices to buffer rows, so the skip
    11	layers' FlashMLA calls read HBM. Entries keep their position, multiplicity
    12	and -1, so the kernel reads the same bytes in the same order.
    13	
    14	This step's own tokens are written by the skip layer itself, after the copy,
    15	so they cannot be staged: their slots map to reserved rows that the skip
    16	layer's KV write fills (row ``gather_rows() + j`` for token j).
    17	
    18	One buffer set serves every group: group g's copy forks from the main stream
    19	after anchor g, when group g-1's skip layers have finished reading.
    20	"""
    21	
    22	import os
    23	from functools import cache
    24	from typing import Any
    25	
    26	import torch
    27	
    28	from vllm.forward_context import get_forward_context
    29	from vllm.model_executor.models.utils import extract_layer_index
    30	from vllm.triton_utils import tl, triton
    31	
    32	TOPK = 2048
    33	MAX_QUERIES = 64  # the most queries a step can stage (one reserved row each)
    34	RESERVED_ROWS = 64
    35	ROW_BYTES = 656
    36	STAGE_BUFFERS = 3  # skip layers per anchor (index_topk_freq - 1)
    37	GATHER_CTAS = 8  # per skip layer
    38	
    39	
    40	@cache
    41	def limits() -> tuple[int, int]:
    42	    """(queries, index columns) the staging buffers are sized for: a decode
    43	    step's tokens (max_num_seqs x (1 + speculative tokens), at most
    44	    MAX_QUERIES) and, under DCP, the decode index width (each staged row is one
    45	    distinct index entry, so queries x columns rows cover any step). Cached:
    46	    the planner's call at load time (config set) fixes what the stager built
    47	    at the first forward uses; with no config, the worst case."""
    48	    import vllm.envs as envs
    49	    from vllm.config import vllm as vllm_config
    50	
    51	    config = vllm_config._current_vllm_config
    52	    if config is None:
    53	        return MAX_QUERIES, TOPK
    54	    spec = config.speculative_config
    55	    queries = config.scheduler_config.max_num_seqs * (
    56	        1 + (spec.num_speculative_tokens if spec is not None else 0)
    57	    )
    58	    columns = TOPK
    59	    if config.parallel_config.decode_context_parallel_size > 1:
    60	        columns = min(envs.VLLM_DCP_SPARSE_DECODE_WIDTH or TOPK, TOPK)
    61	    return min(max(queries, 1), MAX_QUERIES), columns
    62	
    63	
    64	def gather_rows() -> int:
    65	    queries, columns = limits()
    66	    return queries * columns
    67	
    68	
    69	def staging_bytes() -> int:
    70	    return STAGE_BUFFERS * (gather_rows() + RESERVED_ROWS) * ROW_BYTES
    71	
    72	
    73	# A hot shared counter serializes: ~600 winners of one atomic_add took ~20 us
    74	# per group in served decode. Slots are instead marked without atomics, then
    75	# compacted with one counter add per CTA; which row a slot gets does not change
    76	# what any entry reads.
    77	
    78	
    79	@triton.jit
    80	def _mark_rows_kernel(
    81	    idx_ptr, n, new_ptr, row_of_ptr, T: tl.constexpr, BLOCK: tl.constexpr
    82	):
    83	    """Mark every selected slot that is not this step's (idempotent stores)."""
    84	    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    85	    s = tl.load(idx_ptr + offs, mask=offs < n, other=-1)
    86	    is_new = s < 0
    87	    for j in tl.static_range(T):
    88	        nj = tl.load(new_ptr + j)
    89	        is_new |= (s == nj) & (nj >= 0)
    90	    want = ~is_new
    91	    tl.store(row_of_ptr + tl.where(want, s, 0).to(tl.int64), -2, mask=want)
    92	
    93	
    94	@triton.jit
    95	def _compact_rows_kernel(
    96	    row_of_ptr, num_slots, slots_ptr, count_ptr, CHUNK: tl.constexpr
    97	):
    98	    """Give each marked slot of this CTA's range a buffer row."""
    99	    offs = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
   100	    marked = tl.load(row_of_ptr + offs, mask=offs < num_slots, other=-1) == -2
   101	    local = tl.cumsum(marked.to(tl.int32), axis=0)
   102	    base = tl.atomic_add(count_ptr, tl.sum(marked.to(tl.int32), axis=0))
   103	    row = base + local - 1
   104	    tl.store(row_of_ptr + offs, row, mask=marked)
   105	    tl.store(slots_ptr + row, offs, mask=marked)
   106	
   107	
   108	@triton.jit
   109	def _remap_rows_kernel(
   110	    idx_ptr,
   111	    n,
   112	    new_ptr,
   113	    row_of_ptr,
   114	    out_ptr,
   115	    T: tl.constexpr,
   116	    RESERVED: tl.constexpr,
   117	    BLOCK: tl.constexpr,
   118	):
   119	    """Each entry's buffer row: staged row, reserved row (this step's token),
   120	    or -1."""
   121	    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
   122	    m = offs < n
   123	    s = tl.load(idx_ptr + offs, mask=m, other=-1)
   124	    row = tl.load(
   125	        row_of_ptr + tl.where(s >= 0, s, 0).to(tl.int64), mask=m & (s >= 0), other=-1
   126	    )
   127	    for j in tl.static_range(T):
   128	        nj = tl.load(new_ptr + j)
   129	        row = tl.where((s == nj) & (nj >= 0), RESERVED + j, row)
   130	    tl.store(out_ptr + offs, tl.where(s < 0, -1, row), mask=m)
   131	
   132	
   133	@triton.jit
   134	def _copy_rows(src_ptr, dst_ptr, rows, slots, live, ROW_WORDS, BLOCK_W: tl.constexpr):
   135	    words = tl.arange(0, BLOCK_W)
   136	    mask = live[:, None] & (words < ROW_WORDS)[None, :]
   137	    v = tl.load(src_ptr + slots[:, None] * ROW_WORDS + words[None, :], mask=mask)
   138	    dst = dst_ptr + rows[:, None].to(tl.int64) * ROW_WORDS + words[None, :]
   139	    tl.store(dst, v, mask=mask)
   140	
   141	
   142	@triton.jit
   143	def _gather_rows_kernel(
   144	    src0,
   145	    src1,
   146	    src2,
   147	    dst_ptr,
   148	    slots_ptr,
   149	    count_ptr,
   150	    row_of_ptr,
   151	    DST_WORDS: tl.constexpr,
   152	    ROW_WORDS: tl.constexpr,
   153	    BLOCK_R: tl.constexpr,
   154	    BLOCK_W: tl.constexpr,
   155	):
   156	    """dst[layer] row r = src[layer] row slots[r] for r < count, CTAs striding
   157	    over the rows; layer 0's CTAs also clear the claims for the next group."""
   158	    layer = tl.program_id(1)
   159	    count = tl.load(count_ptr)
   160	    dst = dst_ptr + layer.to(tl.int64) * DST_WORDS
   161	    for start in range(tl.program_id(0) * BLOCK_R, count, tl.num_programs(0) * BLOCK_R):
   162	        rows = start + tl.arange(0, BLOCK_R)
   163	        live = rows < count
   164	        slots = tl.load(slots_ptr + rows, mask=live, other=0).to(tl.int64)
   165	        if layer == 0:
   166	            _copy_rows(src0, dst, rows, slots, live, ROW_WORDS, BLOCK_W)
   167	            tl.store(row_of_ptr + slots, -1, mask=live)
   168	        elif layer == 1:
   169	            _copy_rows(src1, dst, rows, slots, live, ROW_WORDS, BLOCK_W)
   170	        else:
   171	            _copy_rows(src2, dst, rows, slots, live, ROW_WORDS, BLOCK_W)
   172	
   173	
   174	class SkipKVStager:
   175	    """Per-rank staging state shared by the attention layers on the rank."""
   176	
   177	    def __init__(self, device: torch.device, layers: dict[int, Any]):
   178	        self.max_queries, self.columns = limits()
   179	        self.gather_rows = gather_rows()
   180	        rows = self.gather_rows + RESERVED_ROWS
   181	        self.buffers = torch.zeros(
   182	            STAGE_BUFFERS, rows // 64, 64, ROW_BYTES, dtype=torch.uint8, device=device
   183	        )
   184	        self.slots = torch.zeros(self.gather_rows, dtype=torch.int32, device=device)
   185	        self.count = torch.zeros(1, dtype=torch.int32, device=device)
   186	        self.rows = torch.full(
   187	            (self.max_queries, self.columns), -1, dtype=torch.int32, device=device
   188	        )
   189	        self.stream = torch.cuda.Stream(device)
   190	        # anchor cache -> (its skip layers' caches, ready event, group number);
   191	        # skip cache -> buffer position
   192	        self.groups: dict[int, tuple[list[torch.Tensor], torch.cuda.Event, int]]
   193	        self.groups = {}
   194	        self.position: dict[int, int] = {}
   195	        self._ready: torch.cuda.Event | None = None
   196	        self._width = self.columns  # the anchor's index width, set by stage()
   197	        self._reserved = self.gather_rows + torch.arange(
   198	            self.max_queries, dtype=torch.int64, device=device
   199	        )
   200	        self.reserved_map = torch.full_like(self._reserved, -1)
   201	        num_slots = 0
   202	        for index, layer in layers.items():
   203	            if layer.impl._reuses_topk:
   204	                continue
   205	            group: list[torch.Tensor] = []
   206	            while index + 1 + len(group) in layers and (
   207	                layers[index + 1 + len(group)].impl._reuses_topk
   208	            ):
   209	                skip = layers[index + 1 + len(group)].kv_cache
   210	                self.position[skip.data_ptr()] = len(group)
   211	                group.append(skip)
   212	                num_slots = skip.numel() // ROW_BYTES
   213	            assert len(group) <= STAGE_BUFFERS
   214	            if group:
   215	                self.groups[layer.kv_cache.data_ptr()] = (
   216	                    group,
   217	                    torch.cuda.Event(),
   218	                    len(self.groups),
   219	                )
   220	        # slot -> staged buffer row, -2 while marked, -1 otherwise (the last
   221	        # entry is unused padding)
   222	        self.row_of = torch.full((num_slots + 1,), -1, dtype=torch.int32, device=device)
   223	        self.row_of[num_slots] = -3
   224	        # VLLM_SKIP_KV_STATS: per group, a histogram of staged rows in 256-row bins
   225	        self.hist = (
   226	            torch.zeros(len(self.groups), self.gather_rows // 256 + 1, device=device)
   227	            if os.environ.get("VLLM_SKIP_KV_STATS")
   228	            else None
   229	        )
   230	
   231	    def stage(
   232	        self, anchor_cache: torch.Tensor, indices: torch.Tensor, new_slots: torch.Tensor
   233	    ) -> None:
   234	        """Anchor, after its FlashMLA launch: start its group's copy."""
   235	        entry = self.groups.get(anchor_cache.data_ptr())
   236	        if entry is None:
   237	            return
   238	        group, ready, number = entry
   239	        t, n = indices.shape[0], indices.numel()
   240	        # every entry may be a distinct row: the buffers must hold them all
   241	        assert t <= self.max_queries and indices.shape[1] <= self.columns, (
   242	            f"skip-KV staging sized for {self.max_queries} x {self.columns}, "
   243	            f"got {tuple(indices.shape)}"
   244	        )
   245	        self._width = indices.shape[1]
   246	        new = new_slots[:t]
   247	        if number == 0:  # this step's tokens are the same for every group
   248	            self.reserved_map[:t] = torch.where(new >= 0, self._reserved[:t], -1)
   249	        self.stream.wait_stream(torch.cuda.current_stream())
   250	        with torch.cuda.stream(self.stream):
   251	            self.count.zero_()
   252	            grid = (triton.cdiv(n, 1024),)
   253	            _mark_rows_kernel[grid](indices, n, new, self.row_of, T=t, BLOCK=1024)
   254	            num_slots = self.row_of.numel() - 1
   255	            _compact_rows_kernel[(triton.cdiv(num_slots, 4096),)](
   256	                self.row_of, num_slots, self.slots, self.count, CHUNK=4096
   257	            )
   258	            _remap_rows_kernel[grid](
   259	                indices,
   260	                n,
   261	                new,
   262	                self.row_of,
   263	                self.rows,
   264	                T=t,
   265	                RESERVED=self.gather_rows,
   266	                BLOCK=1024,
   267	            )
   268	            srcs = [c.view(torch.int32) for c in group]
   269	            srcs += [srcs[0]] * (STAGE_BUFFERS - len(srcs))
   270	            words = ROW_BYTES // 4
   271	            _gather_rows_kernel[(GATHER_CTAS, len(group))](
   272	                *srcs,
   273	                self.buffers.view(torch.int32),
   274	                self.slots,
   275	                self.count,
   276	                self.row_of,
   277	                DST_WORDS=(self.gather_rows + RESERVED_ROWS) * words,
   278	                ROW_WORDS=words,
   279	                BLOCK_R=32,
   280	                BLOCK_W=triton.next_power_of_2(words),
   281	            )
   282	            if self.hist is not None:
   283	                self.hist[number].index_add_(
   284	                    0, (self.count // 256).long(), self.count.new_ones(1).float()
   285	                )
   286	            ready.record(self.stream)
   287	        self._ready = ready
   288	
   289	    def staged(
   290	        self, cache: torch.Tensor, num_queries: int
   291	    ) -> tuple[torch.Tensor, torch.Tensor] | None:
   292	        """Skip layer: its staged buffer and remapped indices, else None."""
   293	        p = self.position.get(cache.data_ptr())
   294	        if p is None:
   295	            return None
   296	        assert self._ready is not None
   297	        torch.cuda.current_stream().wait_event(self._ready)
   298	        rows = self.rows.view(-1)[: num_queries * self._width]
   299	        return self.buffers[p], rows.view(num_queries, self._width)
   300	
   301	    def write_new(
   302	        self,
   303	        cache: torch.Tensor,
   304	        kv_c_normed: torch.Tensor,
   305	        k_pe: torch.Tensor,
   306	        slot_mapping: torch.Tensor,
   307	        k_scale: torch.Tensor,
   308	    ) -> None:
   309	        """Skip layer's KV write: this step's tokens into its reserved rows
   310	        (the mapping is built by the anchor's ``stage``)."""
   311	        p = self.position.get(cache.data_ptr())
   312	        if p is None:
   313	            return
   314	        from vllm import _custom_ops as ops
   315	
   316	        ops.concat_and_cache_mla(
   317	            kv_c_normed,
   318	            k_pe.squeeze(1),
   319	            self.buffers[p],
   320	            self.reserved_map[: slot_mapping.numel()],
   321	            kv_cache_dtype="fp8_ds_mla",
   322	            scale=k_scale,
   323	        )
   324	
   325	
   326	_STAGER: SkipKVStager | None = None
   327	
   328	
   329	def get_stager(device: torch.device) -> SkipKVStager:
   330	    """The rank's stager, built from the bound attention layers on first use."""
   331	    global _STAGER
   332	    if _STAGER is None:
   333	        layers = {
   334	            extract_layer_index(name): layer
   335	            for name, layer in get_forward_context().no_compile_layers.items()
   336	            if hasattr(getattr(layer, "impl", None), "_reuses_topk")
   337	        }
   338	        _STAGER = SkipKVStager(device, layers)
   339	    return _STAGER
```
