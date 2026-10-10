Follow-up on your td_v56 review (same thread context may be gone, so the kernel is re-attached below). Do NOT run commands; reason from the material.

# New evidence (stress mode: one case, fresh random inputs every call, 20,000 calls per run, one GPU each)
Failure = output differs from fp32 reference; "garbage" = values 1e28..1e37 or NaN.
- 64 tokens, every token routed to the SAME 8 cold experts ("same_cold"), shared expert ON, no padding: 13 garbage calls / 20k (another node: 4 / 20k). Highest rate.
- 64 tokens, "cold" (each token 8 of 12 cold experts), shared ON, padding on or off: 4-6 / 20k garbage.
- Same 64-token cold cases with shared expert OFF (sw13/sw2 null): 0 / 40k.
- 64 tokens, shared ON, no routed work at all ("shared_only"), or routed experts unmapped: no garbage (only single-element rel errors 0.0054-0.0070 vs 5e-3 threshold; fp32-atomic ordering noise).
- 48 tokens cold/shared: failures seen (1-5 / 20k in 'cold' with padding); 48 same_cold: 0 so far; 40 tokens cold: 0 / 20k; 31 and 33 tokens all-cold(sparse): 0 / 20k.
- td_v55 (MAX_TOKENS 32) at 32 and 24 tokens same_cold / all-cold: 0 / 20k each (control).
- After every call, y, y13 and y13s are verified zero over their full extents (also before the call): always clean, including around failing calls. So the garbage is produced inside the failing call, not inherited.
- Garbage patterns:
  (a) every token (48..64 of 64 rows) x every one of the 6144 output columns garbage (most common);
  (b) every token row bad but only a few output columns, e.g. cols {4616,4617,4620,4621} or {4494,4495} or {4360..4365} or {5640..5647} (all tokens).
  Pattern (a) looks like the shared expert's whole contribution is garbage (y13s/x2s corrupted, e.g. an S0 unit consuming stale INT4-weight bytes as bf16); pattern (b) looks like a few weight rows of one S1 (or R1) unit being garbage for all tokens.
- With "same_cold" the hot queue holds only the shared expert's groups (32 S0 + 24 S1), so ~116 hot CTAs run shared units and then steal the cold queue (64 entries x (8 R0 + 12 R1 groups)); the 16 cold CTAs run cold units then steal shared work. So shared units and cold routed units interleave in the same CTA rings.

# Question
Find the mechanism. My current suspicion: a consumer reads a stage whose new contents have not landed (its full barrier phase completed early or the wrong phase was waited), or a stage is refilled while a consumer still reads it, in CTAs whose ring mixes shared units (activation area W_BYTES .. W_BYTES + T*128, which for T > 32 extends past S_BYTES into the routed activation-row area) with routed cold units. Check the tx-byte accounting per kind, the phase/parity bookkeeping in the consumer (especially the R0 multi-chunk loop that consumes nch stages without reading descriptors, the shared 'last' path with consumer_sync, hand_off), the producer's empty waits, and anything else that can make a shared unit read stale bytes only when T > 32 AND routed cold work is interleaved. Also consider ordering between bulk copies (async proxy) and generic accesses to the same smem bytes across consecutive uses of a stage by different kinds. Rank hypotheses, and for each give a cheap discriminating experiment (kernel define / debug check) I can run.

# td_v56.cu (numbered)
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
    60	  #define TD_MAX_TOKENS 32  // tokens per call: 32 or 64
    61	#endif
    62	#ifndef TD_COLD_CTAS
    63	  #define TD_COLD_CTAS 16
    64	#endif
    65	constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
    66	constexpr int MAX_TOK = 8, MAX_TOKENS = TD_MAX_TOKENS, MAX_ROUTES = MAX_TOKENS * TOPK,
    67	              MAX_LIST = MAX_ROUTES;
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
   176	// ---------------------------------------------------------------- PTX helpers
   177	__device__ __forceinline__ uint32_t smem_u32(const void* p) {
   178	  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
   179	}
   180	__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
   181	  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
   182	               "r"(count));
   183	}
   184	__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
   185	  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
   186	                   smem_u32(bar)),
   187	               "r"(bytes)
   188	               : "memory");
   189	}
   190	__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
   191	  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
   192	               : "memory");
   193	}
   194	__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
   195	  uint32_t v;
   196	  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
   197	  return v;
   198	}
   199	__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
   200	  uint4 v;
   201	  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
   202	               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
   203	               : "r"(a));
   204	  return v;
   205	}
   206	__device__ __forceinline__ void mbar_wait_a(uint32_t bar, uint32_t parity) {
   207	  asm volatile(
   208	      "{\n .reg .pred p;\n WAITA_%=:\n "
   209	      "mbarrier.try_wait.parity.shared::cta.b64 "
   210	      "p, [%0], %1;\n"
   211	      " @!p bra WAITA_%=;\n}\n" ::"r"(bar),
   212	      "r"(parity)
   213	      : "memory");
   214	}
   215	__device__ __forceinline__ void mbar_arrive_a(uint32_t bar) {
   216	  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
   217	               : "memory");
   218	}
   219	__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
   220	  asm volatile(
   221	      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
   222	      "p, [%0], %1;\n"
   223	      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
   224	      "r"(parity)
   225	      : "memory");
   226	}
   227	__device__ __forceinline__ void bulk_g2s(void* dst, const void* src,
   228	                                         uint32_t bytes, uint64_t* bar) {
   229	  asm volatile(
   230	      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
   231	      "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
   232	      "l"(src), "r"(bytes), "r"(smem_u32(bar))
   233	      : "memory");
   234	}
   235	__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map,
   236	                                       int c0, int c1, int c2, uint64_t* bar) {
   237	  asm volatile(
   238	      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
   239	      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
   240	      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
   241	      "r"(smem_u32(bar))
   242	      : "memory");
   243	}
   244	__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map,
   245	                                       int c0, int c1, uint64_t* bar) {
   246	  asm volatile(
   247	      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
   248	      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
   249	      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
   250	      : "memory");
   251	}
   252	__device__ __forceinline__ void ldmatrix_x4(uint32_t* a, const void* p) {
   253	  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
   254	               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
   255	               : "r"(smem_u32(p)));
   256	}
   257	__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a,
   258	                                         uint32_t b0, uint32_t b1) {
   259	  asm volatile(
   260	      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
   261	      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
   262	      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
   263	      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
   264	}
   265	__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
   266	                                        uint32_t b0, uint32_t b1,
   267	                                        const float* c) {
   268	  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
   269	      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
   270	      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
   271	      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]),
   272	        "f"(c[1]), "f"(c[2]), "f"(c[3]));
   273	}
   274	__device__ __forceinline__ uint32_t prmt0(uint32_t a, uint32_t sel) {
   275	  uint32_t out;
   276	  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(out) : "r"(a), "r"(sel));
   277	  return out;
   278	}
   279	// One Marlin word -> the four f16x2 A registers. Nibbles, low to high:
   280	// (g,k0) (g,k8) (g+8,k0) (g+8,k8) (g,k1) (g,k9) (g+8,k1) (g+8,k9). Low and high
   281	// nibbles become e5m2 bytes (sign at 7, exponent at 3..2, mantissa at 1), and
   282	// an e5m2 byte is the high byte of the f16 it equals.
   283	__device__ __forceinline__ void decode(uint32_t w, uint32_t* a) {
   284	  const uint32_t lo = ((w << 4) & 0x80808080u) | ((w << 1) & 0x0E0E0E0Eu);
   285	  const uint32_t hi = (w & 0x80808080u) | ((w >> 3) & 0x0E0E0E0Eu);
   286	  a[0] = prmt0(lo, 0x2404);  // row g,   k0 k1
   287	  a[1] = prmt0(lo, 0x3414);  // row g+8, k0 k1
   288	  a[2] = prmt0(hi, 0x2404);  // row g,   k8 k9
   289	  a[3] = prmt0(hi, 0x3414);  // row g+8, k8 k9
   290	}
   291	// The same word as symmetric INT4 (code - 8), in the same fragment order: each
   292	// nibble pair lands under 0x6400 as 1024 + code, and one exact fma gives
   293	// (code - 8) * 2^-14, the scale the MXFP4 decode leaves its values at too.
   294	__device__ __forceinline__ void decode_int4(uint32_t w, uint32_t* a) {
   295	  const __half2 unit = __halves2half2(__ushort_as_half(0x0400),
   296	                                      __ushort_as_half(0x0400));  // 2^-14
   297	  const __half2 bias =
   298	      __halves2half2(__ushort_as_half(0xAC08),
   299	                     __ushort_as_half(0xAC08));  // -1032 * 2^-14
   300	  const int shift[4] = {0, 8, 4, 12};  // rows g, g+8 at k0 k1; then at k8 k9
   301	#pragma unroll
   302	  for (int i = 0; i < 4; ++i) {
   303	    const uint32_t t = ((w >> shift[i]) & 0x000F000Fu) | 0x64006400u;
   304	    const __half2 v =
   305	        __hfma2(*reinterpret_cast<const __half2*>(&t), unit, bias);
   306	    a[i] = *reinterpret_cast<const uint32_t*>(&v);
   307	  }
   308	}
   309	// The same fragment order with one shift and four lop3 per word: nibbles
   310	// (0,4) / (2,6) under 0x6400 give 1024 + code, nibbles (1,5) / (3,7) give
   311	// 1024 + 16 * code; one exact hfma2 each brings both to (code - 8) * 2^-14.
   312	__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t mask,
   313	                                                uint32_t magic) {
   314	  uint32_t d;
   315	  asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
   316	      : "=r"(d)
   317	      : "r"(a), "r"(mask), "r"(magic));
   318	  return d;  // (a & mask) | magic
   319	}
   320	__device__ __forceinline__ void decode_int4_fast(uint32_t w, uint32_t* a) {
   321	  const uint32_t magic = 0x64006400u;
   322	  const uint32_t w8 = w >> 8;
   323	  const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, magic);   // rows g,   k0 k1
   324	  const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, magic);  // rows g+8, k0 k1
   325	  const uint32_t hi0 = lop3_and_or(w, 0x00F000F0u, magic);   // rows g,   k8 k9
   326	  const uint32_t hi1 = lop3_and_or(w8, 0x00F000F0u, magic);  // rows g+8, k8 k9
   327	  const __half2 unit =
   328	      __halves2half2(__ushort_as_half(0x0400), __ushort_as_half(0x0400));
   329	  const __half2 bias =
   330	      __halves2half2(__ushort_as_half(0xAC08), __ushort_as_half(0xAC08));
   331	  // (1024 + 16 c) * 2^-18 - 72 * 2^-14 = (c - 8) * 2^-14
   332	  const __half2 unit16 =
   333	      __halves2half2(__ushort_as_half(0x0040), __ushort_as_half(0x0040));
   334	  const __half2 bias16 =
   335	      __halves2half2(__ushort_as_half(0x9C80), __ushort_as_half(0x9C80));
   336	  __half2 v;
   337	  v = __hfma2(*reinterpret_cast<const __half2*>(&lo0), unit, bias);
   338	  a[0] = *reinterpret_cast<const uint32_t*>(&v);
   339	  v = __hfma2(*reinterpret_cast<const __half2*>(&lo1), unit, bias);
   340	  a[1] = *reinterpret_cast<const uint32_t*>(&v);
   341	  v = __hfma2(*reinterpret_cast<const __half2*>(&hi0), unit16, bias16);
   342	  a[2] = *reinterpret_cast<const uint32_t*>(&v);
   343	  v = __hfma2(*reinterpret_cast<const __half2*>(&hi1), unit16, bias16);
   344	  a[3] = *reinterpret_cast<const uint32_t*>(&v);
   345	}
   346	// Programmatic dependent launch: every kernel of a layer is launched early and
   347	// waits here before touching what its predecessor writes; it releases its own
   348	// successor right after, so each launch and prologue hides behind the previous
   349	// kernel instead of following it.
   350	__device__ __forceinline__ void pdl_wait() {
   351	  asm volatile("griddepcontrol.wait;" ::: "memory");
   352	}
   353	__device__ __forceinline__ void pdl_release() {
   354	  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
   355	}
   356	__device__ __forceinline__ float e8m0(uint32_t byte) {
   357	  return __uint_as_float(byte << 23);
   358	}
   359	
   360	// Element k of an activation row -> its slot in the f16 B-fragment layout:
   361	// [k/32 group][tq][k16 block within group][b0 lo, b0 hi, b1 lo, b1 hi].
   362	__device__ __forceinline__ int frag_slot(int k) {
   363	  const int kb = k / 16, r = k % 16, tq = (r % 8) / 2, reg = r / 8, h = r % 2;
   364	  return (((kb / 2) * 4 + tq) * 2 + kb % 2) * 4 + reg * 2 + h;
   365	}
   366	
   367	// Scale a row so its largest magnitude is 2^13 or below and store it in f16;
   368	// the power of two (times the weights' 2^14) goes to *scale.
   369	__device__ __forceinline__ float row_scale(float m, float* scale) {
   370	  const int t = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
   371	  *scale = exp2f(static_cast<float>(t + PLACEMENT_EXP));
   372	  return exp2f(static_cast<float>(-t));
   373	}
   374	
   375	__device__ __forceinline__ float block_max(float m, float* red) {
   376	  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   377	  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
   378	  __syncthreads();
   379	  if (threadIdx.x < 32) {
   380	    m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
   381	    for (int o = 16; o; o >>= 1)
   382	      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   383	    if (threadIdx.x == 0) red[0] = m;
   384	  }
   385	  __syncthreads();
   386	  m = red[0];
   387	  __syncthreads();
   388	  return m;
   389	}
   390	
   391	// ---------------------------------------------------------------- 1: route +
   392	// prep Blocks [0, T): token rows -> f16 fragments. Block T: which GPU runs
   393	// each active expert (below), then this GPU's per-tier expert lists.
   394	//
   395	// Replica assignment. A hot expert runs on its primary GPU; an active cold
   396	// expert with a replica may run on either holder, and every GPU derives the
   397	// same choice from the same router output. The choice balances predicted
   398	// layer time: COST_US is this path's measured graph-replay time per layer
   399	// (GH200, MiMo-V2 shapes) with h hot and c cold experts on a GPU, made
   400	// non-decreasing. Path reversal moves one cold expert at a time off the
   401	// slowest GPU, along at most three replica hops (each hop is a different GPU
   402	// pair, so the flexible experts are tracked as counts per pair), to the end
   403	// that stays fastest, while that end stays below the source's time.
   404	constexpr int EP = 4, PAIRS = EP * (EP - 1) / 2, MAX_EXPERTS = 512,
   405	              MAX_REVERSALS = 64, COST_HOT = 25, COST_COLD = 7;
   406	__constant__ unsigned short COST_US[COST_HOT][COST_COLD] = {
   407	    {0, 62, 111, 161, 226, 269, 305},    {20, 64, 113, 163, 228, 271, 307},
   408	    {30, 65, 115, 165, 229, 272, 308},   {39, 67, 115, 165, 229, 273, 308},
   409	    {45, 67, 115, 165, 230, 273, 309},   {52, 67, 115, 165, 230, 273, 309},
   410	    {58, 69, 116, 165, 230, 273, 309},   {66, 74, 116, 166, 230, 273, 309},
   411	    {73, 84, 116, 166, 231, 273, 309},   {82, 90, 117, 166, 231, 273, 309},
   412	    {87, 102, 118, 166, 231, 273, 309},  {92, 112, 126, 167, 231, 274, 309},
   413	    {100, 120, 128, 167, 231, 274, 309}, {108, 128, 136, 169, 231, 274, 309},
   414	    {116, 139, 151, 169, 231, 274, 309}, {126, 147, 161, 175, 231, 274, 309},
   415	    {136, 155, 171, 180, 232, 275, 309}, {142, 159, 180, 187, 232, 275, 309},
   416	    {147, 164, 189, 193, 233, 275, 309}, {156, 173, 198, 204, 233, 276, 310},
   417	    {164, 181, 207, 215, 233, 277, 310}, {172, 192, 217, 225, 236, 277, 311},
   418	    {180, 203, 226, 235, 239, 278, 312}, {188, 213, 236, 245, 245, 279, 313},
   419	    {196, 224, 246, 254, 254, 279, 314},
   420	};
   421	
   422	struct Placement {
   423	  const int* primary;      // [E] GPU holding the expert's own copy
   424	  const int* secondary;    // [E] GPU holding a cold replica, or -1
   425	  const int* primary_hot;  // [E] 1 when the own copy is in the hot tier
   426	  int num_experts;         // 0: the slot maps already say what runs here
   427	  int ep_rank;
   428	  bool schedule;  // false: every expert runs on its primary
   429	};
   430	
   431	__device__ __forceinline__ int pair_index(int lo, int hi) {
   432	  return lo * EP - lo * (lo + 1) / 2 + (hi - lo - 1);
   433	}
   434	
   435	__device__ __forceinline__ int cost(const unsigned short (*tab)[COST_COLD],
   436	                                    int h, int c) {
   437	  return tab[min(h, COST_HOT - 1)][min(c, COST_COLD - 1)] +
   438	         8 * max(h - (COST_HOT - 1), 0) + 40 * max(c - (COST_COLD - 1), 0);
   439	}
   440	
   441	// One warp. at_low[k] of the total[k] flexible experts of pair k sit on the
   442	// pair's lower GPU; on return (lane 0) at_low holds the balanced split. Each
   443	// iteration, lane l < 15 tries the l-th path from the slowest GPU in the
   444	// reference order (length, then ascending GPU ids): its target, and the pairs
   445	// it needs an edge on the lower (need_lo) or upper (need_hi) GPU of.
   446	__device__ void reverse_paths(const unsigned short (*tab)[COST_COLD],
   447	                              const int* hot, const int* fixed,
   448	                              const int* total, int* at_low) {
   449	  const int lane = threadIdx.x % 32;
   450	  int split[PAIRS], tot[PAIRS];
   451	#pragma unroll
   452	  for (int k = 0; k < PAIRS; ++k) {
   453	    split[k] = at_low[k];
   454	    tot[k] = total[k];
   455	  }
   456	  int h[EP], f[EP];
   457	#pragma unroll
   458	  for (int r = 0; r < EP; ++r) {
   459	    h[r] = hot[r];
   460	    f[r] = fixed[r];
   461	  }
   462	  for (int step = 0; step < MAX_REVERSALS; ++step) {
   463	    int now[EP], after[EP];
   464	#pragma unroll
   465	    for (int r = 0; r < EP; ++r) {
   466	      int c = f[r];
   467	#pragma unroll
   468	      for (int o = 0; o < EP; ++o)
   469	        if (o != r) {
   470	          const int k = pair_index(min(r, o), max(r, o));
   471	          c += r < o ? split[k] : tot[k] - split[k];
   472	        }
   473	      now[r] = cost(tab, h[r], c);
   474	      after[r] = cost(tab, h[r], c + 1);
   475	    }
   476	    int src = 0;
   477	#pragma unroll
   478	    for (int r = 1; r < EP; ++r)
   479	      if (now[r] > now[src]) src = r;
   480	    // this lane's path; the i-th GPU other than src, ascending, is other(i)
   481	    auto other = [src](int i) { return i < src ? i : i + 1; };
   482	    int hop1 = -1, hop2 = -1, hop3 = -1, len = 0;
   483	    if (lane < 3) {
   484	      len = 1;
   485	      hop1 = other(lane);
   486	    } else if (lane < 15) {
   487	      const int m = lane < 9 ? lane - 3 : lane - 9;
   488	      const int first = m / 2, second = m % 2 < first ? m % 2 : m % 2 + 1;
   489	      len = lane < 9 ? 2 : 3;
   490	      hop1 = other(first);
   491	      hop2 = other(second);
   492	      hop3 = other(3 - first - second);
   493	    }
   494	    const int target = len == 1 ? hop1 : len == 2 ? hop2 : hop3;
   495	    unsigned need_lo = 0, need_hi = 0;
   496	    auto need = [&](int u, int v) {
   497	      const unsigned bit = 1u << pair_index(min(u, v), max(u, v));
   498	      if (u < v)
   499	        need_lo |= bit;
   500	      else
   501	        need_hi |= bit;
   502	    };
   503	    if (len >= 1) need(src, hop1);
   504	    if (len >= 2) need(hop1, hop2);
   505	    if (len >= 3) need(hop2, hop3);
   506	    unsigned have_lo = 0, have_hi = 0;
   507	#pragma unroll
   508	    for (int k = 0; k < PAIRS; ++k) {
   509	      have_lo |= (split[k] > 0 ? 1u : 0u) << k;
   510	      have_hi |= (tot[k] - split[k] > 0 ? 1u : 0u) << k;
   511	    }
   512	    int end_v = 0, src_v = 0;
   513	#pragma unroll
   514	    for (int r = 0; r < EP; ++r) {
   515	      if (r == target) end_v = after[r];
   516	      if (r == src) src_v = now[r];
   517	    }
   518	    const bool usable = len > 0 && (need_lo & ~have_lo) == 0 &&
   519	                        (need_hi & ~have_hi) == 0 && end_v < src_v;
   520	    const unsigned key =
   521	        usable ? static_cast<unsigned>(end_v) << 5 | lane : 0xffffffffu;
   522	    const unsigned best = __reduce_min_sync(0xffffffffu, key);
   523	    if (best == 0xffffffffu) break;
   524	    const int from = best & 31;
   525	    need_lo = __shfl_sync(0xffffffffu, need_lo, from);
   526	    need_hi = __shfl_sync(0xffffffffu, need_hi, from);
   527	#pragma unroll
   528	    for (int k = 0; k < PAIRS; ++k)
   529	      split[k] += ((need_hi >> k) & 1) - ((need_lo >> k) & 1);
   530	  }
   531	  if (lane == 0)
   532	#pragma unroll
   533	    for (int k = 0; k < PAIRS; ++k) at_low[k] = split[k];
   534	}
   535	
   536	template <typename IdT>
   537	__global__ void route_prep_kernel(Workspace* ws, const __nv_bfloat16* x,
   538	                                  const IdT* topk_ids, const bool* padding,
   539	                                  const float* topk_weights, const int* hot_map,
   540	                                  const int* cold_map, Placement pl,
   541	                                  int num_tokens, int hot_size, int cold_size,
   542	                                  int num_experts, float routed_scale) {
   543	  // [hot_size + cold_size] each: tier slot -> its routes here, its first entry
   544	  extern __shared__ int n_of[];
   545	  int* base_of = n_of + hot_size + cold_size;
   546	  __shared__ float red[32];
   547	  pdl_wait();
   548	  pdl_release();
   549	  if (blockIdx.x < num_tokens) {
   550	    // every element loads before any is used: these small kernels are latency
   551	    // bound
   552	    constexpr int PER = HIDDEN / PREP_THREADS;
   553	    const int t = blockIdx.x;
   554	    const __nv_bfloat16* __restrict__ xr = x + static_cast<size_t>(t) * HIDDEN;
   555	    float v[PER], m = 0.f;
   556	#pragma unroll
   557	    for (int i = 0; i < PER; ++i)
   558	      v[i] = __bfloat162float(xr[threadIdx.x + i * PREP_THREADS]);
   559	#pragma unroll
   560	    for (int i = 0; i < PER; ++i) m = fmaxf(m, fabsf(v[i]));
   561	    const float inv = row_scale(block_max(m, red), &ws->xs13[t]);
   562	    __half* __restrict__ out = ws->x13 + static_cast<size_t>(t) * HIDDEN;
   563	#pragma unroll
   564	    for (int i = 0; i < PER; ++i)
   565	      out[frag_slot(threadIdx.x + i * PREP_THREADS)] =
   566	          __float2half_rn(v[i] * inv);
   567	    __nv_bfloat16* __restrict__ outb =
   568	        ws->x13b + static_cast<size_t>(t) * HIDDEN;
   569	#pragma unroll
   570	    for (int i = 0; i < PER; ++i)
   571	      outb[frag_slot(threadIdx.x + i * PREP_THREADS)] =
   572	          xr[threadIdx.x + i * PREP_THREADS];
   573	    return;
   574	  }
   575	  __shared__ int count[2];
   576	  for (int i = threadIdx.x; i < 2 * MAX_LIST; i += blockDim.x)
   577	    (&ws->done13[0][0])[i] = 0;
   578	  // per expert: 1 when routed, then (tier << 16 | slot) where it runs here
   579	  __shared__ int state[MAX_EXPERTS];
   580	  __shared__ int hot_n[EP], fixed_n[EP], total[PAIRS], at_low[PAIRS];
   581	  __shared__ int warp_n[PREP_THREADS / 32][PAIRS];
   582	  __shared__ unsigned short tab[COST_HOT][COST_COLD];
   583	  const int E = pl.num_experts, routes = num_tokens * TOPK, r = threadIdx.x;
   584	  for (int i = threadIdx.x; i < hot_size + cold_size; i += blockDim.x)
   585	    n_of[i] = 0;
   586	  if (threadIdx.x < 2) count[threadIdx.x] = 0;
   587	  if (r < MAX_ROUTES) ws->live[r] = 0;
   588	  // every global read is issued up front: this block is a latency chain
   589	  int e = -1;
   590	  float wt = 0.f;
   591	  if (r < routes) {
   592	    e = static_cast<int>(topk_ids[r]);
   593	    // a padded token's routes are dropped, as if its ids were -1
   594	    if (padding != nullptr && padding[r / TOPK]) e = -1;
   595	    wt = topk_weights[r] * routed_scale;
   596	    if (e >= (E > 0 ? E : num_experts)) e = -1;
   597	  }
   598	  // without a placement the slot maps are read with the ids, not after them
   599	  __shared__ int maps[2 * MAX_EXPERTS];
   600	  if (E == 0)
   601	    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
   602	      maps[i] = hot_map[i];
   603	      maps[MAX_EXPERTS + i] = cold_map[i];
   604	    }
   605	  int tier = -1, local = -1, slot = -1;
   606	  if (E > 0) {
   607	    const int x_e = threadIdx.x;
   608	    int p = -1, s = -1, hm = -1, cm = -1;
   609	    bool hot = false;
   610	    if (x_e < E) {
   611	      p = pl.primary[x_e];
   612	      hot = pl.primary_hot[x_e] != 0;
   613	      s = pl.secondary[x_e];
   614	      hm = hot_map[x_e];
   615	      cm = cold_map[x_e];
   616	      state[x_e] = 0;
   617	    }
   618	    if (x_e < EP) hot_n[x_e] = fixed_n[x_e] = 0;
   619	    if (x_e < PAIRS) total[x_e] = at_low[x_e] = 0;
   620	    if (x_e < COST_HOT * COST_COLD)
   621	      tab[x_e / COST_COLD][x_e % COST_COLD] =
   622	          COST_US[x_e / COST_COLD][x_e % COST_COLD];
   623	    __syncthreads();
   624	    if (e >= 0) state[e] = 1;
   625	    __syncthreads();
   626	    int k = -1;
   627	    if (x_e < E && state[x_e]) {
   628	      if (hot)
   629	        atomicAdd(&hot_n[p], 1);
   630	      else if (s < 0)
   631	        atomicAdd(&fixed_n[p], 1);
   632	      else {
   633	        k = pair_index(min(p, s), max(p, s));
   634	        atomicAdd(&total[k], 1);
   635	        if (p < s) atomicAdd(&at_low[k], 1);
   636	      }
   637	    } else {
   638	      p = -1;
   639	    }
   640	    // a flexible expert's position among its pair's, in ascending expert id
   641	    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
   642	    int pos = 0;
   643	#pragma unroll
   644	    for (int j = 0; j < PAIRS; ++j) {
   645	      const unsigned ballot = __ballot_sync(0xffffffffu, k == j);
   646	      if (k == j) pos = __popc(ballot & ((1u << lane) - 1));
   647	      if (lane == 0) warp_n[warp][j] = __popc(ballot);
   648	    }
   649	    __syncthreads();
   650	    if (k >= 0)
   651	      for (int w = 0; w < warp; ++w) pos += warp_n[w][k];
   652	    if (threadIdx.x < 32 && pl.schedule)
   653	      reverse_paths(tab, hot_n, fixed_n, total, at_low);
   654	    __syncthreads();
   655	    if (p >= 0) {
   656	      const int runs_on = k < 0 ? p : pos < at_low[k] ? min(p, s) : max(p, s);
   657	      int code = -1;
   658	      if (runs_on == pl.ep_rank && hot && hm >= 0)
   659	        code = hm;
   660	      else if (runs_on == pl.ep_rank && !hot && cm >= 0)
   661	        code = 1 << 16 | cm;
   662	      state[x_e] = code;
   663	    }
   664	    __syncthreads();
   665	    if (e >= 0 && state[e] >= 0) {
   666	      tier = state[e] >> 16;
   667	      local = state[e] & 0xffff;
   668	    }
   669	  } else {
   670	    __syncthreads();
   671	    if (e >= 0) {
   672	      const int h = maps[e], c = maps[MAX_EXPERTS + e];
   673	      if (h >= 0) {
   674	        tier = 0;
   675	        local = h;
   676	      } else if (c >= 0) {
   677	        tier = 1;
   678	        local = c;
   679	      }
   680	    }
   681	  }
   682	  // each route's position among its expert's; the first route then opens
   683	  // ceil(n / MAX_TOK) consecutive entries, MAX_TOK routes each
   684	  const int idx = tier * hot_size + local;
   685	  int pos = -1;
   686	  if (tier >= 0) pos = atomicAdd(&n_of[idx], 1);
   687	  __syncthreads();
   688	  if (pos == 0) {
   689	    const int n = n_of[idx], entries = (n + MAX_TOK - 1) / MAX_TOK;
   690	    const int base = atomicAdd(&count[tier], entries);
   691	    base_of[idx] = base;
   692	    for (int q = 0; q < entries; ++q) {
   693	      Expert& ex = ws->lists[tier][base + q];
   694	      ex.local = local;
   695	      ex.ntok = min(MAX_TOK, n - q * MAX_TOK);
   696	    }
   697	  }
   698	  __syncthreads();
   699	  if (tier >= 0) {
   700	    slot = base_of[idx] + pos / MAX_TOK;
   701	    Expert& ex = ws->lists[tier][slot];
   702	    const int i = pos % MAX_TOK;
   703	    ex.tok[i] = r / TOPK;
   704	    ex.route[i] = r;
   705	    ex.wt[i] = wt;
   706	    ws->live[r] = 1;
   707	  }
   708	  __syncthreads();
   709	  if (threadIdx.x < 2) ws->counts[threadIdx.x] = count[threadIdx.x];
   710	  if (threadIdx.x == 0) {
   711	    ws->done_s = 0;
   712	    ws->next[0] = ws->next[1] = 0;
   713	    ws->T = num_tokens;
   714	    // unsigned wrap; at 0 the ready flags are cleared so that a flag last
   715	    // set 2^32 calls ago cannot alias the new epoch
   716	    int next = static_cast<int>(static_cast<unsigned>(ws->epoch) + 1u);
   717	    if (next == 0) {
   718	      for (int i = 0; i < 2 * MAX_LIST; ++i) (&ws->ready[0][0])[i] = 0;
   719	      ws->ready_s = 0;
   720	      next = 1;
   721	    }
   722	    ws->epoch = next;
   723	  }
   724	}
   725	
   726	// ---------------------------------------------------------------- 2: the
   727	// layer, one persistent CTA per SM. CTAs [0, cold_ctas) run the cold tier,
   728	// the rest the hot tier. Each CTA runs its share of its tier's w13 units, then
   729	// its share of the w2 units. An entry's w13 chunks are counted as they flush;
   730	// the CTA whose flush completes an entry runs silu * up for its routes and
   731	// releases ready[entry]. A w2 unit's producer issues the weights first and
   732	// waits on ready only before loading the activation rows.
   733	__device__ __forceinline__ int cold_ctas_for(int n_hot, int n_cold) {
   734	  if (n_cold == 0) return 0;
   735	  return n_hot == 0 ? GRID : TD_COLD_CTAS;
   736	}
   737	__device__ __forceinline__ int ld_acquire(const int* p) {
   738	  int v;
   739	  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
   740	               : "=r"(v)
   741	               : "l"(p)
   742	               : "memory");
   743	  return v;
   744	}
   745	__device__ __forceinline__ void st_release(int* p, int v) {
   746	  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v)
   747	               : "memory");
   748	}
   749	__device__ __forceinline__ void fence_proxy_async() {
   750	  asm volatile("fence.proxy.async.global;" ::: "memory");
   751	}
   752	// predicated fire-and-forget add: no branch per element
   753	__device__ __forceinline__ void red_add_if(float* a, float v, bool p) {
   754	  asm volatile(
   755	      "{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q red.global.add.f32 [%0], "
   756	      "%1;\n}" ::"l"(a),
   757	      "f"(v), "r"(static_cast<int>(p))
   758	      : "memory");
   759	}
   760	__device__ __forceinline__ void red_add_v4(float* a, float4 v) {
   761	  asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};" ::"l"(a),
   762	               "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
   763	               : "memory");
   764	}
   765	__device__ __forceinline__ void sts_f32(uint32_t a, float v) {
   766	  asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v));
   767	}
   768	__device__ __forceinline__ float4 lds_f4(uint32_t a) {
   769	  float4 v;
   770	  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
   771	               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
   772	               : "r"(a));
   773	  return v;
   774	}
   775	// One warp's 64 rows x ntok tokens of a routed unit (acc * per-token scale)
   776	// added into base + row(token) * stride: staged through the warp's smem
   777	// scratch [token][68] (conflict-free), then vector reds over contiguous rows.
   778	constexpr int SCR_LD = 68;
   779	__device__ __forceinline__ void flush_rows(float (*acc)[4], const float* fs,
   780	                                           uint32_t scr, float* base,
   781	                                           const int* rows4, int stride,
   782	                                           int ntok, int g, int tq, int lane) {
   783	#pragma unroll
   784	  for (int mb = 0; mb < 4; ++mb)
   785	#pragma unroll
   786	    for (int i = 0; i < 4; ++i) {
   787	      sts_f32(
   788	          scr + ((2 * tq + (i & 1)) * SCR_LD + mb * 16 + g + (i >> 1) * 8) * 4,
   789	          acc[mb][i] * fs[i & 1]);
   790	      acc[mb][i] = 0.f;
   791	    }
   792	  __syncwarp();
   793	#pragma unroll
   794	  for (int j = 0; j < MAX_TOK / 2; ++j) {  // token c = lane / 16 + 2 j
   795	    const int k = lane + 32 * j;
   796	    if (k < ntok * 16) {
   797	      const int c = k >> 4, r4 = k & 15;
   798	      const float4 v = lds_f4(scr + (c * SCR_LD + r4 * 4) * 4);
   799	      red_add_v4(base + static_cast<size_t>(rows4[j]) * stride + r4 * 4, v);
   800	    }
   801	  }
   802	  __syncwarp();
   803	}
   804	__device__ __forceinline__ void consumer_sync() {
   805	  asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
   806	}
   807	
   808	// silu(gate) * up for one entry's routes, one warp per route: f16 fragments
   809	// of x2 with the row's power-of-two scale; the y13 rows are zeroed for the
   810	// next call.
   811	__device__ __forceinline__ void activate_route(Workspace* ws, int r, int lane) {
   812	  float* __restrict__ yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
   813	  float a[INTER / 32], m = 0.f;
   814	#pragma unroll
   815	  for (int q = 0; q < INTER / 32; ++q) {
   816	    const float g = __ldcg(yr + q * 32 + lane),
   817	                u = __ldcg(yr + INTER + q * 32 + lane);
   818	    a[q] = __fdividef(g, 1.f + __expf(-g)) * u;
   819	    m = fmaxf(m, fabsf(a[q]));
   820	  }
   821	  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
   822	  float scale;
   823	  const float inv = row_scale(m, &scale);
   824	  if (lane == 0) ws->xs2[r] = scale;
   825	  __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
   826	  if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
   827	#pragma unroll
   828	  for (int q = 0; q < INTER / 32; ++q) {
   829	    out[frag_slot(q * 32 + lane)] = __float2half_rn(a[q] * inv);
   830	    yr[q * 32 + lane] = 0.f;
   831	    yr[INTER + q * 32 + lane] = 0.f;
   832	  }
   833	}
   834	
   835	// activate_route for the n routes in lanes [0, n) of routel, K at a time
   836	template <int K>
   837	__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
   838	                                                int n, int lane,
   839	                                                uint64_t* tl = nullptr) {
   840	  for (int r0 = 0; r0 < n; r0 += K) {
   841	    float a[K][INTER / 32], m[K];
   842	    float* yr[K];
   843	#pragma unroll
   844	    for (int j = 0; j < K; ++j) {
   845	      const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
   846	      m[j] = 0.f;
   847	      if (r0 + j < n) {
   848	        yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
   849	#pragma unroll
   850	        for (int q = 0; q < INTER / 32; ++q) {
   851	          const float g = __ldcg(yr[j] + q * 32 + lane),
   852	                      u = __ldcg(yr[j] + INTER + q * 32 + lane);
   853	          a[j][q] = __fdividef(g, 1.f + __expf(-g)) * u;
   854	          m[j] = fmaxf(m[j], fabsf(a[j][q]));
   855	        }
   856	      }
   857	    }
   858	#pragma unroll
   859	    for (int j = 0; j < K; ++j) {
   860	      if (r0 + j >= n) break;
   861	      for (int o = 16; o; o >>= 1)
   862	        m[j] = fmaxf(m[j], __shfl_xor_sync(0xffffffffu, m[j], o));
   863	      float scale;
   864	      const float inv = row_scale(m[j], &scale);
   865	      const int r = __shfl_sync(0xffffffffu, routel, r0 + j);
   866	      if (lane == 0) ws->xs2[r] = scale;
   867	      __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
   868	      if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
   869	#pragma unroll
   870	      for (int q = 0; q < INTER / 32; ++q) {
   871	        out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
   872	      }
   873	    }
   874	  }
   875	}
   876	
   877	// zero the y13 rows of the n routes in lanes [0, n) of routel for the next
   878	// call
   879	__device__ __forceinline__ void zero_routes(Workspace* ws, int routel, int n,
   880	                                            int lane) {
   881	  for (int j = 0; j < n; ++j) {
   882	    const int r = __shfl_sync(0xffffffffu, routel, j);
   883	    float4* __restrict__ yr =
   884	        reinterpret_cast<float4*>(ws->y13 + static_cast<size_t>(r) * 2 * INTER);
   885	#pragma unroll
   886	    for (int q = 0; q < 2 * INTER / 128; ++q)
   887	      yr[q * 32 + lane] = make_float4(0.f, 0.f, 0.f, 0.f);
   888	  }
   889	}
   890	
   891	// The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
   892	// y13s is zeroed for the next call.
   893	__device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
   894	                                                int lane) {
   895	  for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
   896	    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
   897	    __nv_bfloat16* __restrict__ out =
   898	        ws->x2s + static_cast<size_t>(tok) * INTER;
   899	#pragma unroll
   900	    for (int q = 0; q < INTER / 32; ++q) {
   901	      const int k = q * 32 + lane;
   902	      const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
   903	      out[frag_slot(k)] =
   904	          __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
   905	      yr[k] = 0.f;
   906	      yr[INTER + k] = 0.f;
   907	    }
   908	  }
   909	}
   910	
   911	// One routed unit's 16 group steps for this warp. PH 0 (w13): 4 row blocks x
   912	// 2 K halves of a 64-row x 1024 box; PH 1 (w2): 8 row blocks of a 128-row x
   913	// 512 box. All smem offsets are immediates off two per-warp bases.
   914	// One routed unit for this warp: all 4 row blocks of a 64-row tile over 4
   915	// group steps of K, so each lane's 16 B of a Marlin k16 row (its 4 row blocks)
   916	// is one conflict-free LDS.128, and each activation fragment feeds 4 MMAs.
   917	// w13 unit (64 rows x 1024): warp w takes K steps [4w, 4w + 4); w2 unit (128
   918	// rows x 512): warp w takes 64-row half w / 4, K steps [4 (w % 4), + 4).
   919	template <int PH>
   920	__device__ __forceinline__ void consume_routed(uint32_t wb, uint32_t sb,
   921	                                               uint32_t xb, float (*acc)[4]) {
   922	  constexpr int KROW = PH ? 1024 : 512;  // bytes per k16 row of the box
   923	  constexpr int SROW = PH ? 256 : 128;   // bytes per scale group row
   924	#pragma unroll
   925	  for (int j = 0; j < 4; ++j) {
   926	    const uint4 w0 = lds_v4(wb + (2 * j) * KROW);
   927	    const uint4 w1 = lds_v4(wb + (2 * j + 1) * KROW);
   928	    const uint4 sw = lds_v4(sb + j * SROW);
   929	    const uint4 xv = lds_v4(xb + j * 64);
   930	    const uint32_t w0s[4] = {w0.x, w0.y, w0.z, w0.w},
   931	                   w1s[4] = {w1.x, w1.y, w1.z, w1.w};
   932	    const uint32_t sws[4] = {sw.x, sw.y, sw.z, sw.w};
   933	    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
   934	#pragma unroll
   935	    for (int mb = 0; mb < 4; mb += 2) {
   936	      float d0[4], d1[4];
   937	      uint32_t a0[4], a1[4];
   938	      decode_int4_fast(w0s[mb], a0);
   939	      decode_int4_fast(w0s[mb + 1], a1);
   940	      mma_f16(d0, a0, xv.x, xv.y, zero);
   941	      mma_f16(d1, a1, xv.x, xv.y, zero);
   942	      decode_int4_fast(w1s[mb], a0);
   943	      decode_int4_fast(w1s[mb + 1], a1);
   944	      mma_f16(d0, a0, xv.z, xv.w, d0);
   945	      mma_f16(d1, a1, xv.z, xv.w, d1);
   946	      const float s00 = __uint_as_float(sws[mb] << 16),
   947	                  s01 = __uint_as_float(sws[mb] & 0xFFFF0000u),
   948	                  s10 = __uint_as_float(sws[mb + 1] << 16),
   949	                  s11 = __uint_as_float(sws[mb + 1] & 0xFFFF0000u);
   950	      acc[mb][0] = fmaf(s00, d0[0], acc[mb][0]);
   951	      acc[mb][1] = fmaf(s00, d0[1], acc[mb][1]);
   952	      acc[mb][2] = fmaf(s01, d0[2], acc[mb][2]);
   953	      acc[mb][3] = fmaf(s01, d0[3], acc[mb][3]);
   954	      acc[mb + 1][0] = fmaf(s10, d1[0], acc[mb + 1][0]);
   955	      acc[mb + 1][1] = fmaf(s10, d1[1], acc[mb + 1][1]);
   956	      acc[mb + 1][2] = fmaf(s11, d1[2], acc[mb + 1][2]);
   957	      acc[mb + 1][3] = fmaf(s11, d1[3], acc[mb + 1][3]);
   958	    }
   959	  }
   960	}
   961	
   962	// Work queues. Each tier's work is a list of groups, claimed in order with
   963	// one atomic by whichever CTA's producer is free: the shared expert's w13
   964	// (tile x 12 chunks), the routed w13 (entry x tile, 6 chunks), the shared w2
   965	// (tile, 8 chunks), the routed w2 (entry x 128-row unit). An entry's w13
   966	// groups go out early and to many CTAs at once, so its w2 units rarely wait;
   967	// the queue ends on 1-unit groups, so CTAs finish within a unit of each
   968	// other. A CTA whose own tier's queue is empty takes the other tier's.
   969	enum Kind : int { K_S0 = 0, K_R0 = 1, K_S1 = 2, K_R1 = 3, K_END = 15 };
   970	constexpr int SCH0 = 12;               // shared w13 chunks per group
   971	constexpr int SG0 = CHUNKS_S0 / SCH0;  // groups per shared w13 tile
   972	static_assert(CHUNKS_S0 % SCH0 == 0, "shared w13 groups tile K");
   973	struct Group {
   974	  int kind, x, c0,
   975	      nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
   976	};
   977	// one claimed group, decoded, with its entry record (scheduler -> producer)
   978	struct SchedEntry {
   979	  int kind, x, c0, nch, q, ei, t0, ntok, local, ready;
   980	  int tok[MAX_TOK], route[MAX_TOK];
   981	  float f[MAX_TOK];  // w13: the token's x13 scale; w2: the route weight
   982	};
   983	// routed w2 units per claim (consecutive tiles of one entry): GR1 above
   984	// GR1_T tokens, else 1. Larger claims amortize the producer's claim, record
   985	// and ready round trips (M=16/32 -5..7%); at M=8 they unbalance the tail.
   986	#ifndef TD_GR1
   987	  #define TD_GR1 2
   988	#endif
   989	#ifndef TD_GR1_T
   990	  #define TD_GR1_T 8
   991	#endif
   992	constexpr int GR1 = TD_GR1;
   993	static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
   994	// A routed w13 tile's 12 K chunks go out as R0S adjacent groups, so several
   995	// CTAs share a late tile and its entry's activation is not stuck behind one
   996	// CTA's 12 serial units (the w2 producers spin on it).
   997	#ifndef TD_R0S
   998	  #define TD_R0S 1  // 2-4 measured slower: more w13 flushes
   999	#endif
  1000	constexpr int R0S = TD_R0S, R0CH = CHUNKS0 / R0S;
  1001	static_assert(CHUNKS0 % R0S == 0, "w13 groups split the K chunks evenly");
  1002	// warps 1..7 run at most STAGES units ahead of warp 0 (the ring), so with
  1003	// more units than that between w13 flushes no warp can arrive at the handoff
  1004	// barrier for the next flush before warp 0 has synced on this one
  1005	static_assert(R0CH > STAGES, "w13 handoff barrier generations cannot overlap");
  1006	__device__ __forceinline__ int queue_len(int n, bool sh, int g1) {
  1007	  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
  1008	         n * (TILES0 * R0S + TILES1 / g1);
  1009	}
  1010	__device__ __forceinline__ Group group_at(int gi, int n, bool sh, int g1) {
  1011	  const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0 * R0S,
  1012	            ns1 = sh ? TILES_S1 : 0;
  1013	  if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
  1014	  gi -= ns0;
  1015	  if (gi < nr0) return {K_R0, gi / R0S, (gi % R0S) * R0CH, R0CH};
  1016	  gi -= nr0;
  1017	  if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
  1018	  gi -= ns1;
  1019	  return {K_R1, gi, 0, g1};
  1020	}
  1021	
  1022	__global__ void __launch_bounds__(THREADS, 1)
  1023	    layer_kernel(const __grid_constant__ Params p) {
  1024	  extern __shared__ __align__(128) unsigned char smem[];
  1025	  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  1026	  uint64_t* empty = full + STAGES;
  1027	  // 1 KB aligned (128 B swizzle) by an offset into smem, so every load off it
  1028	  // stays a shared-space LDS rather than a generic LD
  1029	  unsigned char* ring =
  1030	      smem + SMEM_HEAD +
  1031	      ((1024u - ((smem_u32(smem) + SMEM_HEAD) & 1023u)) & 1023u);
  1032	  __shared__ int4 desc[STAGES];
  1033	  // per stage, routed units: each token's destination row (w13: its route's
  1034	  // y13 row; w2: its token's y row) and the scale its flush applies
  1035	  // (w13: xs13[token]; w2: the route weight, times the row's xs2 at the
  1036	  // flush), written by the producer
  1037	  __shared__ int sd_row[STAGES][MAX_TOK];
  1038	  __shared__ float sd_f[STAGES][MAX_TOK];
  1039	  __shared__ int s_last;
  1040	  __shared__ SchedEntry sq[TD_SQ];
  1041	  __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
  1042	  __shared__ int4 hq_info[TD_HQ];  // (q, entry, chunks, end)
  1043	  __shared__ uint64_t hq_full[TD_HQ], hq_empty[TD_HQ];
  1044	  __shared__ __align__(16) float scr_all[CONSUMER_WARPS][MAX_TOK * SCR_LD];
  1045	  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  1046	  Workspace* ws = p.ws;
  1047	  if (threadIdx.x == 0) {
  1048	    for (int s = 0; s < STAGES; ++s) {
  1049	      mbar_init(&full[s], 1);
  1050	      mbar_init(&empty[s], CONSUMER_WARPS);
  1051	    }
  1052	    for (int k = 0; k < TD_SQ; ++k) {
  1053	      mbar_init(&sq_full[k], 1);
  1054	      mbar_init(&sq_empty[k], 1);
  1055	    }
  1056	    for (int k = 0; k < TD_HQ; ++k) {
  1057	      mbar_init(&hq_full[k], CONSUMER_WARPS);
  1058	      mbar_init(&hq_empty[k], 1);
  1059	    }
  1060	    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  1061	  }
  1062	  __syncthreads();
  1063	  pdl_wait();  // route_prep's lists, rows and counters
  1064	  pdl_release();
  1065	  const int n_tier[2] = {ws->counts[0], ws->counts[1]};
  1066	  const int epoch = ws->epoch, T = ws->T;
  1067	  const bool sh = p.has_shared;
  1068	  const int g1 = T > TD_GR1_T ? GR1 : 1;
  1069	  const int len[2] = {queue_len(n_tier[0], sh, g1),
  1070	                      queue_len(n_tier[1], false, g1)};
  1071	  const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
  1072	  const int own = static_cast<int>(blockIdx.x) < cold_ctas ? 1 : 0;
  1073	
  1074	  if (warp == CONSUMER_WARPS + 2) {  // finisher warp
  1075	    int k = 0;
  1076	    uint32_t kph = 0;
  1077	    while (true) {
  1078	      if (lane == 0) mbar_wait(&hq_full[k], kph);
  1079	      __syncwarp();
  1080	      const int4 h = hq_info[k];
  1081	      __syncwarp();
  1082	      if (lane == 0) mbar_arrive(&hq_empty[k]);
  1083	      if (++k == TD_HQ) {
  1084	        k = 0;
  1085	        kph ^= 1;
  1086	      }
  1087	      if (h.w) break;
  1088	      const int q = h.x, ei = h.y, nch = h.z;
  1089	      const Expert& e = ws->lists[q][ei];
  1090	      const int n = e.ntok, routel = e.route[lane & (MAX_TOK - 1)];
  1091	      int done = 0;
  1092	      if (lane == 0) {
  1093	        int old;
  1094	        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
  1095	                     : "=r"(old)
  1096	                     : "l"(&ws->done13[q][ei]), "r"(nch)
  1097	                     : "memory");
  1098	        done = old + nch == UNITS0;
  1099	      }
  1100	      done = __shfl_sync(0xffffffffu, done, 0);
  1101	      if (done) {
  1102	        __syncwarp();  // lane 0's acquire (the count) ordered before every
  1103	                       // lane's y13 reads
  1104	        activate_routes<TD_ACTK>(ws, routel, n, lane);
  1105	        __syncwarp();
  1106	        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
  1107	        zero_routes(ws, routel, n, lane);
  1108	      }
  1109	    }
  1110	    return;
  1111	  }
  1112	  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp
  1113	    // token t's x13 scale in lane t % 32 of register t / 32
  1114	    float xs13r[MAX_TOKENS / 32];
  1115	#pragma unroll
  1116	    for (int j = 0; j < MAX_TOKENS / 32; ++j)
  1117	      xs13r[j] = lane + 32 * j < T ? ws->xs13[lane + 32 * j] : 0.f;
  1118	    int q = own, n = 0, k = 0;
  1119	    bool stolen = false;
  1120	    int gi = 0;  // this group's claim; the next one is issued before its loads
  1121	    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
  1122	    gi = __shfl_sync(0xffffffffu, gi, 0);
  1123	    while (true) {
  1124	      if (n >= TD_SQ) {  // slot k is free once its last group is issued
  1125	        if (lane == 0) mbar_wait(&sq_empty[k], ((n / TD_SQ) - 1) & 1);
  1126	        __syncwarp();
  1127	      }
  1128	      SchedEntry& e = sq[k];
  1129	      int gn = 0;
  1130	      // no claim-ahead past a w13 or shared w2 group: its successor is
  1131	      // claimed once the producer has issued all of its chunks
  1132	      const bool heavy =
  1133	          gi < (q ? len[1] : len[0]) &&
  1134	          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1).kind !=
  1135	              K_R1;
  1136	      if (lane == 0 && !heavy) gn = atomicAdd(&ws->next[q], 1);
  1137	      if (gi >= (q ? len[1] : len[0])) {
  1138	        bool steal = !stolen;
  1139	        if (steal) {
  1140	          stolen = true;
  1141	          q ^= 1;
  1142	          if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
  1143	          gi = __shfl_sync(0xffffffffu, gi, 0);
  1144	          continue;
  1145	        }
  1146	        if (lane == 0) e.kind = K_END;
  1147	        __syncwarp();
  1148	        if (lane == 0) mbar_arrive(&sq_full[k]);
  1149	        break;
  1150	      }
  1151	      const Group gr =
  1152	          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1);
  1153	      int ei = 0, t0 = 0;
  1154	      if (gr.kind == K_R0) {
  1155	        ei = gr.x / TILES0;
  1156	        t0 = gr.x - ei * TILES0;
  1157	      } else if (gr.kind == K_R1) {
  1158	        if (g1 == 1) {
  1159	          ei = gr.x / TILES1;
  1160	          t0 = gr.x - ei * TILES1;
  1161	        } else {
  1162	          ei = gr.x / (TILES1 / GR1);
  1163	          t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;
  1164	        }
  1165	      }
  1166	      int rdy = 0;
  1167	      if (gr.kind == K_R0 || gr.kind == K_R1) {
  1168	        // one round trip: every field at once (lanes past ntok read a valid
  1169	        // slot and are masked later)
  1170	        const Expert& x = ws->lists[q][ei];
  1171	        const int l7 = lane & (MAX_TOK - 1);
  1172	        const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
  1173	                  routel = x.route[l7];
  1174	        const float wtl = x.wt[l7];
  1175	        // the ready probe in flight with the record loads
  1176	        if (lane == 0 && gr.kind == K_R1)
  1177	          rdy = ld_acquire(&ws->ready[q][ei]) == epoch;
  1178	        float xs = __shfl_sync(0xffffffffu, xs13r[0], tokl & 31);
  1179	#pragma unroll
  1180	        for (int j = 1; j < MAX_TOKENS / 32; ++j) {
  1181	          const float v = __shfl_sync(0xffffffffu, xs13r[j], tokl & 31);
  1182	          if (tokl >> 5 == j) xs = v;
  1183	        }
  1184	        if (lane < MAX_TOK) {
  1185	          e.tok[lane] = tokl;
  1186	          e.route[lane] = routel;
  1187	          e.f[lane] = gr.kind == K_R0 ? xs : wtl;
  1188	        }
  1189	        if (lane == 0) {
  1190	          e.ntok = ntok;
  1191	          e.local = local;
  1192	        }
  1193	      }
  1194	      if (lane == 0) {
  1195	        e.kind = gr.kind;
  1196	        e.x = gr.x;
  1197	        e.c0 = gr.c0;
  1198	        e.nch = gr.nch;
  1199	        e.q = q;
  1200	        e.ei = ei;
  1201	        e.t0 = t0;
  1202	        // one probe: if the entry is ready, this acquire is handed to the
  1203	        // producer by the FIFO barrier and its spin is skipped
  1204	        e.ready = rdy;
  1205	      }
  1206	      __syncwarp();
  1207	      if (lane == 0) mbar_arrive(&sq_full[k]);
  1208	      if (heavy && lane == 0) {
  1209	        mbar_wait(&sq_empty[k], (n / TD_SQ) & 1);
  1210	        gn = atomicAdd(&ws->next[q], 1);
  1211	      }
  1212	      if (++k == TD_SQ) k = 0;
  1213	      ++n;
  1214	      gi = __shfl_sync(0xffffffffu, gn, 0);
  1215	    }
  1216	    return;
  1217	  }
  1218	  if (warp == CONSUMER_WARPS) {  // producer warp
  1219	    if (lane == 0) {
  1220	      for (int q = 0; q < 2; ++q)
  1221	        for (int k = 0; k < 2; ++k) {
  1222	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1223	                           reinterpret_cast<uint64_t>(&p.tier[q].w[k]))
  1224	                       : "memory");
  1225	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1226	                           reinterpret_cast<uint64_t>(&p.tier[q].s[k]))
  1227	                       : "memory");
  1228	        }
  1229	      if (sh)
  1230	        for (int k = 0; k < 2; ++k)
  1231	          asm volatile("prefetch.tensormap [%0];" ::"l"(
  1232	                           reinterpret_cast<uint64_t>(&p.sw[k]))
  1233	                       : "memory");
  1234	    }
  1235	    int last_ready = -1, it = 0, k = 0;
  1236	    uint32_t kph = 0;
  1237	    const int gi = 0;
  1238	    (void)gi;
  1239	    while (true) {
  1240	      if (lane == 0) mbar_wait(&sq_full[k], kph);
  1241	      __syncwarp();
  1242	      const SchedEntry& e = sq[k];
  1243	      if (e.kind == K_END) break;
  1244	      const Group gr = {e.kind, e.x, e.c0, e.nch};
  1245	      const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = e.ready;
  1246	      int ntok = 0, local = 0, tokl = 0, routel = 0;
  1247	      float fl = 0.f;
  1248	      if (gr.kind == K_R0 || gr.kind == K_R1) {
  1249	        const int l7 = lane & (MAX_TOK - 1);
  1250	        ntok = e.ntok;
  1251	        local = e.local;
  1252	        tokl = e.tok[l7];
  1253	        routel = e.route[l7];
  1254	        fl = e.f[l7];
  1255	      }
  1256	      const Tier& tr = p.tier[q];
  1257	      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
  1258	        const int s = it % STAGES, c = gr.c0 + ci;
  1259	        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
  1260	        if (lane == 0 && it >= STAGES)
  1261	          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
  1262	        __syncwarp();
  1263	        const int hdr = gr.kind | q << 4 | (ci == 0) << 8 |
  1264	                        (ci == gr.nch - 1) << 9 | gr.nch << 16;
  1265	        if (gr.kind == K_R0) {
  1266	          if (lane < ntok) {
  1267	            sd_row[s][lane] = routel;
  1268	            sd_f[s][lane] = fl;
  1269	          }
  1270	          __syncwarp();
  1271	          if (lane == 0) {
  1272	            desc[s] = make_int4(hdr, ei, t0, ntok);
  1273	            mbar_expect_tx(&full[s],
  1274	                           W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
  1275	            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
  1276	            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
  1277	          }
  1278	          __syncwarp();
  1279	          if (lane < ntok && lane < XROWS)
  1280	            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
  1281	                     ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
  1282	                     XROW_BYTES0, &full[s]);
  1283	        } else if (gr.kind == K_R1) {
  1284	          const int t = t0 + ci;
  1285	          if (lane < ntok) {
  1286	            sd_row[s][lane] = tokl;
  1287	            sd_f[s][lane] = fl;
  1288	          }
  1289	          __syncwarp();
  1290	          if (lane == 0) {
  1291	            desc[s] = make_int4(hdr, ei, t, ntok);
  1292	            mbar_expect_tx(&full[s],
  1293	                           W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
  1294	            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
  1295	            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
  1296	          }
  1297	          // weights are in flight; only the activation rows wait for the
  1298	          // entry's ready (every copying lane acquires for itself)
  1299	          if ((ei | q << 16) != last_ready) {
  1300	            if (!pre_ready) {
  1301	              while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
  1302	            }
  1303	            fence_proxy_async();
  1304	            last_ready = ei | q << 16;
  1305	          }
  1306	          __syncwarp();
  1307	          if (lane < ntok && lane < XROWS)
  1308	            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
  1309	                     ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
  1310	                     &full[s]);
  1311	        } else {  // shared expert
  1312	          if (lane == 0) {
  1313	            desc[s] = make_int4(hdr, gr.x, c, 0);
  1314	            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
  1315	            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
  1316	                   &full[s]);
  1317	          }
  1318	          if (gr.kind == K_S1 &&
  1319	              last_ready != -2) {  // every copying lane acquires
  1320	            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
  1321	            fence_proxy_async();
  1322	          }
  1323	          if (gr.kind == K_S1) last_ready = -2;
  1324	          __syncwarp();
  1325	          for (int t = lane; t < T; t += 32)
  1326	            bulk_g2s(dst + W_BYTES + t * XS_BYTES,
  1327	                     (gr.kind == K_S0
  1328	                          ? ws->x13b + static_cast<size_t>(t) * HIDDEN
  1329	                          : ws->x2s + static_cast<size_t>(t) * INTER) +
  1330	                         c * CKS,
  1331	                     XS_BYTES, &full[s]);
  1332	        }
  1333	      }
  1334	      __syncwarp();
  1335	      if (lane == 0) mbar_arrive(&sq_empty[k]);
  1336	      if (++k == TD_SQ) {
  1337	        k = 0;
  1338	        kph ^= 1;
  1339	      }
  1340	    }
  1341	    if (lane == 0) {  // end of work
  1342	      const int s = it % STAGES;
  1343	      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
  1344	      desc[s] = make_int4(K_END, 0, 0, 0);
  1345	      mbar_arrive(&full[s]);
  1346	    }
  1347	    return;
  1348	  }
  1349	
  1350	  // consumer warps
  1351	  const int g = lane / 4, tq = lane % 4;
  1352	  const int nt_n = (T + 7) / 8;
  1353	  float acc[4][4] = {};
  1354	  float accs[2][NT_MAX][4] = {};
  1355	  int hk = 0, hn = 0;  // handoff slot and count (same in every consumer)
  1356	  const auto hand_off = [&](int4 h) {
  1357	    // every warp waits for the slot's previous use to be taken before it
  1358	    // arrives (no arrival may enter a phase whose predecessor nobody waited
  1359	    // on); __syncwarp gathers the warp's reds and warp 0's info for lane 0's
  1360	    // release
  1361	    if (lane == 0) {
  1362	      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
  1363	      if (warp == 0) hq_info[hk] = h;
  1364	    }
  1365	    __syncwarp();
  1366	    if (lane == 0) mbar_arrive(&hq_full[hk]);
  1367	    if (++hk == TD_HQ) hk = 0;
  1368	    ++hn;
  1369	  };
  1370	  // per-warp smem offsets within a stage, computed once
  1371	  const uint32_t ring_u = smem_u32(ring), full_u = smem_u32(full),
  1372	                 empty_u = smem_u32(empty);
  1373	  const uint32_t desc_u = smem_u32(desc), sdf_u = smem_u32(&sd_f[0][2 * tq]);
  1374	  const uint32_t scr_u = smem_u32(scr_all[warp]),
  1375	                 sdr0_u = smem_u32(&sd_row[0][0]);
  1376	  static_assert(CONSUMER_WARPS == 8,
  1377	                "8 K slices (w13), 2 halves x 4 K slices (w2)");
  1378	  const int h1 = warp / 4, k1 = warp % 4;
  1379	  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  1380	  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  1381	  const uint32_t xo1 =
  1382	      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (k1 * 16 + tq) * 16;
  1383	  int s = 0;
  1384	  uint32_t ph = 0;
  1385	  for (int it = 0;; ++it) {
  1386	    mbar_wait_a(full_u + 8 * s, ph);
  1387	    const uint4 du = lds_v4(desc_u + 16 * s);
  1388	    const int4 d = make_int4(du.x, du.y, du.z, du.w);
  1389	    const int kind = d.x & 15;
  1390	    if (kind == K_END) {
  1391	      hand_off(make_int4(0, 0, 0, 1));
  1392	      break;
  1393	    }
  1394	    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
  1395	    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
  1396	    const uint32_t st_u = ring_u + s * STAGE_BYTES;
  1397	    const uint32_t empty_s = empty_u + 8 * s;
  1398	    // this lane's two tokens of a routed unit: destination rows and scales
  1399	    const int ntok = d.w;
  1400	    // flush metadata of stage s (plain shared loads the compiler may schedule
  1401	    // into the math), read before the stage is released
  1402	    const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
  1403	      fs_[0] = sd_f[s_][2 * tq];
  1404	      fs_[1] = sd_f[s_][2 * tq + 1];
  1405	      if (r1) {
  1406	        const unsigned char* xr =
  1407	            ring + static_cast<size_t>(s_) * STAGE_BYTES + W_BYTES + S_BYTES +
  1408	            ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
  1409	        fs_[0] *= *reinterpret_cast<const float*>(xr);
  1410	        fs_[1] *=
  1411	            *reinterpret_cast<const float*>(xr + (XROWS > 1 ? XROW_STRIDE : 0));
  1412	      }
  1413	#pragma unroll
  1414	      for (int j = 0; j < MAX_TOK / 2; ++j)
  1415	        rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
  1416	    };
  1417	    float fs[2];
  1418	    int rows4[MAX_TOK / 2];
  1419	
  1420	    if (kind == K_R0) {
  1421	      // the group's nch chunks in a row: its later stages carry nothing new
  1422	      // for the consumer (same entry, tile, tokens), so no descriptor reads
  1423	      for (int c = 0;; ++c) {
  1424	        const uint32_t su = ring_u + s * STAGE_BYTES;
  1425	        consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
  1426	        if (c == nch - 1) flush_meta(s, fs, rows4, false);
  1427	        __syncwarp();
  1428	        if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
  1429	        if (c == nch - 1) break;
  1430	        if (++s == STAGES) {
  1431	          s = 0;
  1432	          ph ^= 1;
  1433	        }
  1434	        mbar_wait_a(full_u + 8 * s, ph);
  1435	      }
  1436	      {
  1437	        const int ei = d.y, t = d.z;
  1438	        flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
  1439	                   ntok, g, tq, lane);
  1440	        hand_off(make_int4(q, ei, nch, 0));
  1441	      }
  1442	    } else if (kind == K_R1) {
  1443	      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
  1444	      flush_meta(s, fs, rows4, true);
  1445	      __syncwarp();
  1446	      if (lane == 0) mbar_arrive_a(empty_s);
  1447	      const int t = d.z;
  1448	      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
  1449	                 g, tq, lane);
  1450	    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
  1451	      const unsigned char* xa = st + W_BYTES;
  1452	#pragma unroll
  1453	      for (int ks = 0; ks < CKS / 16; ++ks) {
  1454	        uint32_t af[2][4];
  1455	#pragma unroll
  1456	        for (int rb = 0; rb < 2; ++rb) {
  1457	          const int row =
  1458	              warp * 32 + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
  1459	          const int ck = ks * 2 + (lane >> 4);
  1460	          ldmatrix_x4(af[rb], st + row * 128 + ((ck ^ (row & 7)) << 4));
  1461	        }
  1462	#pragma unroll
  1463	        for (int nt = 0; nt < NT_MAX; ++nt) {
  1464	          if (nt < nt_n) {
  1465	            const uint2 bv = *reinterpret_cast<const uint2*>(
  1466	                xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 +
  1467	                (ks & 1) * 8);
  1468	            mma_bf16(accs[0][nt], af[0], bv.x, bv.y);
  1469	            mma_bf16(accs[1][nt], af[1], bv.x, bv.y);
  1470	          }
  1471	        }
  1472	      }
  1473	      __syncwarp();
  1474	      if (lane == 0) mbar_arrive_a(empty_s);
  1475	      if (last) {
  1476	        const int t = d.y;
  1477	#pragma unroll
  1478	        for (int rb = 0; rb < 2; ++rb)
  1479	#pragma unroll
  1480	          for (int nt = 0; nt < NT_MAX; ++nt)
  1481	#pragma unroll
  1482	            for (int i = 0; i < 4; ++i) {
  1483	              const int tok = nt * 8 + 2 * tq + (i & 1);
  1484	              const int row = t * RS + warp * 32 + rb * 16 + g + (i >> 1) * 8;
  1485	              if (nt < nt_n && tok < T) {
  1486	                if (kind == K_S0)
  1487	                  atomicAdd(
  1488	                      &ws->y13s[static_cast<size_t>(tok) * 2 * INTER + row],
  1489	                      accs[rb][nt][i]);
  1490	                else
  1491	                  atomicAdd(&ws->y[static_cast<size_t>(tok) * HIDDEN + row],
  1492	                            p.shared_scale * accs[rb][nt][i]);
  1493	              }
  1494	              accs[rb][nt][i] = 0.f;
  1495	            }
  1496	        if (kind == K_S0) {
  1497	          __threadfence();  // every thread's own y13s atomics, before the count
  1498	          consumer_sync();
  1499	          if (threadIdx.x == 0) {
  1500	            __threadfence();
  1501	            s_last = atomicAdd(&ws->done_s, nch) + nch == UNITS_S0;
  1502	          }
  1503	          consumer_sync();
  1504	          if (s_last) {
  1505	            __threadfence();
  1506	            activate_shared(ws, T, warp, lane);
  1507	            fence_proxy_async();
  1508	            __threadfence();
  1509	            consumer_sync();
  1510	            if (threadIdx.x == 0) st_release(&ws->ready_s, epoch);
  1511	          }
  1512	        }
  1513	      }
  1514	    }
  1515	    if (++s == STAGES) {
  1516	      s = 0;
  1517	      ph ^= 1;
  1518	    }
  1519	  }
  1520	}
  1521	
  1522	// ---------------------------------------------------------------- 5: finalize
  1523	// One float4 per thread: grid [T][HIDDEN / 1024] x 256.
  1524	__global__ void finalize_kernel(Workspace* ws,
  1525	                                __nv_bfloat16* __restrict__ out) {
  1526	  pdl_wait();
  1527	  const size_t i = (static_cast<size_t>(blockIdx.y) * HIDDEN +
  1528	                    blockIdx.x * 1024 + threadIdx.x * 4);
  1529	  float4* y = reinterpret_cast<float4*>(ws->y + i);
  1530	  const float4 v = *y;
  1531	  *y = make_float4(0.f, 0.f, 0.f, 0.f);
  1532	  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + i);
  1533	  o[0] = __floats2bfloat162_rn(v.x, v.y);
  1534	  o[1] = __floats2bfloat162_rn(v.z, v.w);
  1535	}
  1536	
  1537	// ---------------------------------------------------------------- host entry
  1538	// Plain C ABI (no torch headers: the build is the kernel alone), called from
  1539	// Python through ctypes with raw device pointers and the current stream.
  1540	#define TD_REQUIRE(c, msg)                              \
  1541	  do {                                                  \
  1542	    if (!(c)) {                                         \
  1543	      std::fprintf(stderr, "tiered_decode: %s\n", msg); \
  1544	      return -1;                                        \
  1545	    }                                                   \
  1546	  } while (0)
  1547	
  1548	// [n2][n1][n0] contiguous elements of elem bytes; box {b0, b1, 1}
  1549	static int make_map3(CUtensorMap* map, const void* p, uint64_t n0, uint64_t n1,
  1550	                     uint64_t n2, int elem, CUtensorMapDataType type,
  1551	                     uint32_t b0, uint32_t b1) {
  1552	  const cuuint64_t dims[3] = {n0, n1, n2};
  1553	  const cuuint64_t strides[2] = {n0 * elem, n0 * n1 * elem};
  1554	  const cuuint32_t box[3] = {b0, b1, 1}, unit[3] = {1, 1, 1};
  1555	  return cuTensorMapEncodeTiled(
  1556	             map, type, 3, const_cast<void*>(p), dims, strides, box, unit,
  1557	             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
  1558	             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
  1559	             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
  1560	             ? 0
  1561	             : -1;
  1562	}
  1563	// [rows][cols] bf16 with a {b0, b1} box, 128 B swizzled
  1564	static int make_map_2d_sw128(CUtensorMap* map, const void* p, uint64_t cols,
  1565	                             uint64_t rows, uint32_t b0, uint32_t b1) {
  1566	  const cuuint64_t dims[2] = {cols, rows};
  1567	  const cuuint64_t strides[1] = {cols * 2};
  1568	  const cuuint32_t box[2] = {b0, b1}, unit[2] = {1, 1};
  1569	  return cuTensorMapEncodeTiled(
  1570	             map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p),
  1571	             dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE,
  1572	             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
  1573	             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
  1574	             ? 0
  1575	             : -1;
  1576	}
  1577	static int fill_tier(Tier& tr, const void* w13, const void* s13, const void* w2,
  1578	                     const void* s2, int e) {
  1579	  std::memset(&tr, 0, sizeof(tr));
  1580	  if (e == 0) return 0;
  1581	  int r = make_map3(&tr.w[0], w13, 2 * INTER * 2, HIDDEN / 16, e, 4,
  1582	                    CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R0, KT0);
  1583	  r |= make_map3(&tr.w[1], w2, HIDDEN * 2, INTER / 16, e, 4,
  1584	                 CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R1, KT1);
  1585	  r |= make_map3(&tr.s[0], s13, 2 * INTER, HIDDEN / 32, e, 2,
  1586	                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R0, G0);
  1587	  r |= make_map3(&tr.s[1], s2, HIDDEN, INTER / 32, e, 2,
  1588	                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R1, G1);
  1589	  return r;
  1590	}
  1591	
  1592	}  // namespace tiered_decode
  1593	
  1594	using namespace tiered_decode;
  1595	
  1596	extern "C" long long td_workspace_bytes() { return sizeof(Workspace); }
  1597	
  1598	extern "C" int td_forward(void* out, const void* x, const int* ids,
  1599	                          const float* wt, const int* hot_map,
  1600	                          const int* cold_map, int T, const void* hw13,
  1601	                          const void* hs13, const void* hw2, const void* hs2,
  1602	                          int hot_size, const void* cw13, const void* cs13,
  1603	                          const void* cw2, const void* cs2, int cold_size,
  1604	                          void* workspace, int pdl_launch, const bool* padding,
  1605	                          const void* sw13, const void* sw2, float shared_scale,
  1606	                          float routed_scale, int num_experts,
  1607	                          void* stream_ptr) {
  1608	  TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..MAX_TOKENS tokens");
  1609	  TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
  1610	  TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS, "1..512 experts");
  1611	  Params p{};
  1612	  TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
  1613	             "hot tier maps");
  1614	  TD_REQUIRE(fill_tier(p.tier[1], cw13, cs13, cw2, cs2, cold_size) == 0,
  1615	             "cold tier maps");
  1616	  p.ws = reinterpret_cast<Workspace*>(workspace);
  1617	  p.has_shared = sw13 != nullptr;
  1618	  p.shared_scale = shared_scale;
  1619	  if (p.has_shared) {
  1620	    TD_REQUIRE(
  1621	        make_map_2d_sw128(&p.sw[0], sw13, HIDDEN, 2 * INTER, CKS, RS) == 0,
  1622	        "sw13 map");
  1623	    TD_REQUIRE(make_map_2d_sw128(&p.sw[1], sw2, INTER, HIDDEN, CKS, RS) == 0,
  1624	               "sw2 map");
  1625	  }
  1626	  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  1627	  static bool attrs = false;
  1628	  if (!attrs) {
  1629	    cudaFuncSetAttribute(
  1630	        layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  1631	    attrs = true;
  1632	  }
  1633	  cudaLaunchAttribute pdl[1];
  1634	  pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  1635	  pdl[0].val.programmaticStreamSerializationAllowed = 1;
  1636	  auto config = [&](dim3 grid, dim3 block, size_t smem) {
  1637	    cudaLaunchConfig_t c = {};
  1638	    c.gridDim = grid;
  1639	    c.blockDim = block;
  1640	    c.dynamicSmemBytes = smem;
  1641	    c.stream = stream;
  1642	    c.attrs = pdl;
  1643	    c.numAttrs = pdl_launch ? 1 : 0;
  1644	    return c;
  1645	  };
  1646	  Placement pl{};
  1647	  const size_t route_smem =
  1648	      2 * static_cast<size_t>(hot_size + cold_size) * sizeof(int);
  1649	  cudaLaunchConfig_t c = config(dim3(T + 1), dim3(PREP_THREADS), route_smem);
  1650	  if (cudaLaunchKernelEx(&c, route_prep_kernel<int>, p.ws,
  1651	                         reinterpret_cast<const __nv_bfloat16*>(x), ids,
  1652	                         padding, wt, hot_map, cold_map, pl, T, hot_size,
  1653	                         cold_size, num_experts, routed_scale) != cudaSuccess)
  1654	    return -2;
  1655	  c = config(dim3(GRID), dim3(THREADS), SMEM_BYTES);
  1656	  if (cudaLaunchKernelEx(&c, layer_kernel, p) != cudaSuccess) return -3;
  1657	  c = config(dim3(HIDDEN / 1024, T), dim3(256), 0);
  1658	  if (cudaLaunchKernelEx(&c, finalize_kernel, p.ws,
  1659	                         reinterpret_cast<__nv_bfloat16*>(out)) != cudaSuccess)
  1660	    return -4;
  1661	  return 0;
  1662	}
  1663	
  1664	// debug: byte offsets of the buffers that every call must leave zeroed
  1665	extern "C" void td_workspace_offsets(long long* o) {
  1666	  o[0] = offsetof(Workspace, y);
  1667	  o[1] = sizeof(float) * MAX_TOKENS * HIDDEN;
  1668	  o[2] = offsetof(Workspace, y13);
  1669	  o[3] = sizeof(float) * MAX_ROUTES * 2 * INTER;
  1670	  o[4] = offsetof(Workspace, y13s);
  1671	  o[5] = sizeof(float) * MAX_TOKENS * 2 * INTER;
  1672	}
```
