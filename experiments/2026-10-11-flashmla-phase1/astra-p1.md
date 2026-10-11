# Review request: FlashMLA sparse decode Phase 1 (vLLM-built split schedule with early stop + setmaxnreg split)

You are reviewing code; do NOT run commands (the sandbox cannot start on this machine). Everything you need is inlined below. Be adversarial: I want correctness holes, CUDA-graph hazards, unsafe assumptions, and test gaps. Rank findings by severity, cite file:line from the excerpts, and say what test would catch each.

## Context
- GH200 (sm90, 132 SMs). GLM-5.3 (DeepSeek V3.2-style sparse MLA, fp8_ds_mla KV, 656 B/token) served with DCP4. vLLM calls FlashMLA's sm90 sparse fp8 decode with b = 1, s_q = T (all decode tokens of all requests flattened), h_q = 64, index rows of width W (768 in prod via VLLM_DCP_SPARSE_DECODE_WIDTH, else 2048) produced by `triton_filter_and_convert_dcp_index`: each row's slots owned by this DCP rank are compacted to a prefix [0, count) and the tail is -1. `valid_counts[t]` is exact even when it exceeds W (row overflowed the width; the row holds its first W owned slots).
- FlashMLA's planner (get_mla_metadata_kernel, excerpt below) gives each SM part `ceil((nb+5)/parts) + 5` blocks; for b = 1 this leaves splits unbalanced (12 blocks over 2 parts -> 9+3; over 4 parts -> 5+5+2), and it always covers the full width.
- Phase 1 (this change): (1) vLLM builds the DecodingSchedMeta itself on device with a tiny Triton kernel from max(valid_counts) (early stop at the longest live row: blocks past it are -1 in every row so skipping them is exact) and splits the blocks evenly: longest = ceil(nb/parts), splits = ceil(nb/longest), bounds nb*i//splits. (2) FlashMLA register split for the V3.2 h64 kernel changed from setmaxnreg 192/160/152 (WG0 consumer / WG1 / producer) to 208/160/136, which removes ptxas C7512 (all 44 wgmma serialized) -> 3 DEPBARs. Launch-time regs are 168/thread x 384 threads; setmaxnreg only redistributes within the CTA, so the sum must be <= 3*168 = 504 (a 512 sum deadlocked in setmaxnreg.inc during the sweep).
- Measured (graph replay, 78 layers): 0.18-1.06 ms/step saved at T = 8..64. Outputs differ from prod only by fp32 split-boundary reordering (<= 1 bf16 ulp); the register split was bitwise-identical in Phase 0 on its configs.
- Constraint: same math only (no algorithm change). Rows with zero live keys must still produce a non-finite LSE (combine drops +inf/NaN -> -inf) and a finite output.

## FlashMLA diff (branch dcp-decode-regs, on vLLM's pinned FlashMLA a8f794d)
```diff
diff --git a/csrc/sm90/decode/sparse_fp8/config.h b/csrc/sm90/decode/sparse_fp8/config.h
index e5631f3..76334d1 100644
--- a/csrc/sm90/decode/sparse_fp8/config.h
+++ b/csrc/sm90/decode/sparse_fp8/config.h
@@ -33,6 +33,16 @@ static constexpr int BLOCK_M = 64;
 static constexpr int TOPK_BLOCK_SIZE = 64;
 static constexpr int NUM_K_BUFS = 2;
 
+// setmaxnreg split: consumer WG0 / PV WG1 / producer WG2. The sum must stay
+// <= 3 * 168 (the launch-time allocation), since setmaxnreg only moves
+// registers within the CTA. For V3.2 h64, 192/160/152 makes ptxas serialize
+// every wgmma (C7512); 208/160/136 keeps them in flight.
+static constexpr bool TUNED_REGS = MODEL_TYPE == ModelType::V32 && NUM_HEADS == 64;
+static constexpr int NUM_REGS_WG0 = TUNED_REGS ? 208 : 192;
+static constexpr int NUM_REGS_WG1 = 160;
+static constexpr int NUM_REGS_PRODUCER = TUNED_REGS ? 136 : 152;
+static_assert(NUM_REGS_WG0 + NUM_REGS_WG1 + NUM_REGS_PRODUCER <= 3 * 168);
+
 using SmemLayoutQTile = decltype(tile_to_shape(
     GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
     Shape<Int<BLOCK_M>, Int<64>>{}
diff --git a/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh b/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh
index 9994568..9c4b9fd 100644
--- a/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh
+++ b/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh
@@ -184,7 +184,7 @@ __device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnD
     };
 
     if (warpgroup_idx == 0) {
-        cutlass::arch::warpgroup_reg_alloc<192>();
+        cutlass::arch::warpgroup_reg_alloc<NUM_REGS_WG0>();
 
         TiledMMA tiled_mma_QK = TiledMMA_QK{};
         ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
@@ -364,7 +364,7 @@ __device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnD
             sync_all_threads_in_cluster();
         }
     } else if (warpgroup_idx == 1) {
-        cutlass::arch::warpgroup_reg_dealloc<160>();
+        cutlass::arch::warpgroup_reg_dealloc<NUM_REGS_WG1>();
 
         TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
         ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
@@ -453,7 +453,7 @@ __device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnD
         }
     } else {
         // Producer warpgroup
-        cutlass::arch::warpgroup_reg_dealloc<152>();
+        cutlass::arch::warpgroup_reg_dealloc<NUM_REGS_PRODUCER>();
 
         static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
         static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
@@ -752,6 +752,13 @@ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &pa
 
     constexpr size_t smem_size = sizeof(SharedMemoryPlan);
     KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
+    static const int num_regs = [&] {
+        cudaFuncAttributes attr;
+        KU_CUDA_CHECK(cudaFuncGetAttributes(&attr, mla_kernel));
+        return attr.numRegs;
+    }();
+    // setmaxnreg.inc would wait forever for registers the CTA does not own
+    KU_ASSERT(NUM_REGS_WG0 + NUM_REGS_WG1 + NUM_REGS_PRODUCER <= 3 * num_regs);
 
     // NOTE Don't use PDL because of potential compiler bugs!
     // cudaLaunchAttribute mla_kernel_attributes[1];
```

## FlashMLA: how the kernel consumes the schedule (csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh)
```cpp
   136	    int bar_phase_k = 0; // Don't use array here to prevent using local memory
   137	
   138	    // Programmatic Dependent Launch: Wait for the previous kernel to finish
   139	    // Don't use PDL because of compiler bugs!
   140	    // cudaGridDependencySynchronize();
   141	    
   142	    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
   143	
   144	    if (sched_meta.begin_req_idx >= params.b) return;
   145	
   146	    if (warp_idx == 0 && elect_one_sync()) {
   147	        Tensor gQ = flat_divide(
   148	            tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
   149	            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
   150	        )(_, _, head_block_idx, _0{});
   151	        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
   152	        plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
   153	    }
   154	
   155	    ku::barrier_cluster_wait_acquire();
   156	
   157	    struct MainloopArgs {
   158	        int start_block_idx, end_block_idx;
   159	        bool is_no_split;
   160	
   161	        // The following fields are only valid for MODEL1
   162	        int topk_length, extra_topk_length, num_orig_kv_blocks;
   163	    };
   164	    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
   165	        MainloopArgs args;
   166	        int total_topk_padded;
   167	        if constexpr (MODEL_TYPE == ModelType::V32) {
   168	            total_topk_padded = params.topk;
   169	        } else {
   170	            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
   171	            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
   172	            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
   173	            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
   174	            args.topk_length = topk_length;
   175	            args.extra_topk_length = extra_topk_length;
   176	            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
   177	        }
   178	
   179	        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
   180	        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
   181	        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
   182	
   183	        return args;
   184	    };
   185	
   186	    if (warpgroup_idx == 0) {
...
   300	            }
   301	            
   302	            float o_scales[2];
   303	            CUTE_UNROLL
   304	            for (int i = 0; i < 2; ++i) {
   305	                if (args.is_no_split) {
   306	                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
   307	                } else {
   308	                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
   309	                }
   310	                if (idx_in_warpgroup%4 == 0) {
   311	                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
   312	                    plan.sOScale[row] = o_scales[i];
   313	                }
   314	            }
   315	
   316	            // This is a synchronization point for warpgroup 0/1.
   317	            // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
   318	            // Warpgroup 1 should wait wg 0 for sL to be ready
   319	            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);
   320	
   321	            CUTE_UNROLL
   322	            for (int i = 0; i < 2; ++i)
   323	                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
   324	            
   325	            int start_head_idx = head_block_idx*BLOCK_M;
   326	            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
   327	            if (args.is_no_split) {
   328	                bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
   329	                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
   330	                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
   331	                    make_stride(params.stride_o_h_q, _1{})
   332	                ));
   333	                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;	// (BLOCK_M) : (1)
   334	
   335	                store_o<true>(rO, gO, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);
   336	
   337	                int i = threadIdx.x;
   338	                if (i < num_valid_seq_q) {
   339	                    float cur_L = sL[i];
   340	                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
   341	                }
   342	
   343	                cute::tma_store_wait<0>();
   344	            } else {
   345	                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
   346	                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
   347	                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
   348	                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
   349	                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
   350	                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
...
   690	    KU_ASSERT(params.d_qk == HEAD_DIM_K);
   691	    KU_ASSERT(params.d_v == HEAD_DIM_V);
   692	    KU_ASSERT(params.h_q % BLOCK_M == 0);
   693	    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
   694	        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
   695	        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
   696	        if (params.extra_kv != nullptr) {
   697	            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
   698	        }
   699	    } else {
   700	        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
   701	        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
   702	        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
   703	    }
   704	
   705	    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
   706	    auto tma_Q = cute::make_tma_copy(
   707	        SM90_TMA_LOAD{},
   708	        make_tensor(
   709	            make_gmem_ptr((bf16*)params.q),
   710	            make_layout(
   711	                shape_Q,
   712	                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
   713	            )
   714	        ),
   715	        SmemLayoutQ{}
   716	    );
   717	    
   718	    CUtensorMap tensor_map_o;
   719	    {
   720	        // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
   721	        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
   722	        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
   723	        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
   724	        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
   725	        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
   726	            &tensor_map_o,
   727	            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
   728	            5,
   729	            params.out,
   730	            size,
   731	            stride,
   732	            box_size,
   733	            elem_stride,
   734	            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
   735	            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
   736	                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
   737	                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
   738	                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
   739	            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
   740	            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   741	        );
   742	        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
   743	    }
   744	
   745	    TmaParams<
   746	        decltype(shape_Q), decltype(tma_Q)
   747	    > tma_params = {
   748	        shape_Q, tma_Q,
   749	        tensor_map_o
   750	    };
   751	    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;
   752	
   753	    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
   754	    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
   755	    static const int num_regs = [&] {
   756	        cudaFuncAttributes attr;
   757	        KU_CUDA_CHECK(cudaFuncGetAttributes(&attr, mla_kernel));
   758	        return attr.numRegs;
   759	    }();
   760	    // setmaxnreg.inc would wait forever for registers the CTA does not own
```
csrc/params.h:
```cpp
struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Both inclusive
    int begin_block_idx, end_block_idx; // Inclusive, exclusive
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);
```
csrc/api/sparse_decode.h (sm90 get_meta and the schedule/shape handling):
```cpp
    56	    )
    57	
    58	public:
    59	    DecodeImplMeta get_meta(int h_q, int s_q) override {
    60	        Arch arch = Arch();
    61	        return {
    62	            std::max(arch.num_sms / s_q / (h_q/64), 1),
    63	            5,
    64	            64
    65	        };
    66	    }
   420	        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
   421	        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
   422	        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
   423	        get_current_cuda_stream(q)
   424	    };
   425	
   426	    // Get MLA metadata if necessary
   427	    Tensor o_accum, lse_accum;
   428	    if (!tile_scheduler_metadata.has_value()) {
   429	        tile_scheduler_metadata = torch::stable::new_empty(q, {impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, ScalarType::Int);
   430	        num_splits = torch::stable::new_empty(q, {b+1}, ScalarType::Int);
   431	        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
   432	        KU_CHECK_CONTIGUOUS(num_splits);
   433	
   434	        GetDecodeSchedMetaParams get_sched_meta_params = {
   435	            b, s_q,
   436	            impl_meta.block_size_topk,
   437	            impl_meta.fixed_overhead_num_blocks,
   438	            topk,
   439	            extra_topk,
   440	            ku::get_optional_tensor_ptr<int>(topk_length),
   441	            ku::get_optional_tensor_ptr<int>(extra_topk_length),
   442	            nullptr,
   443	            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
   444	            num_splits->mutable_data_ptr<int>(),
   445	            impl_meta.num_sm_parts,
   446	            get_current_cuda_stream(q)
   447	        };
   448	        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
   449	    }
   450	    // Stick the metadata pointers to `params`
   451	    KU_CHECK_DEVICE(tile_scheduler_metadata);
   452	    KU_CHECK_DEVICE(num_splits);
   453	    KU_CHECK_DTYPE(tile_scheduler_metadata, ScalarType::Int);
   454	    KU_CHECK_DTYPE(num_splits, ScalarType::Int);
   455	    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
   456	    KU_CHECK_CONTIGUOUS(num_splits);
   457	    KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
   458	    KU_CHECK_SHAPE(num_splits, b+1);
   459	    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
   460	    params.num_splits_ptr = num_splits->mutable_data_ptr<int>();
   461	    params.num_sm_parts = impl_meta.num_sm_parts;
   462	
   463	    // Allocate intermediate buffers for split-KV
   464	    const int total_num_splits = b + impl_meta.num_sm_parts;
   465	    lse_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q}, ScalarType::Float);
   466	    o_accum = torch::stable::new_empty(q, {total_num_splits, s_q, h_q, d_v}, ScalarType::Float);
   467	    KU_CHECK_CONTIGUOUS(lse_accum);
   468	    KU_CHECK_CONTIGUOUS(o_accum);
   469	    params.lse_accum = lse_accum.mutable_data_ptr<float>();
   470	    params.o_accum = o_accum.mutable_data_ptr<float>();
```
csrc/smxx/decode/combine/combine.cu (excerpt):
```cpp
    25	    const int s_q_idx = blockIdx.y;
    26	    const int h_block_idx = blockIdx.z;
    27	    const int warp_idx = threadIdx.x / 32;
    28	    const int lane_idx = threadIdx.x % 32;
    29	
    30	    int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M*h_block_idx);
    31	    if (warp_idx >= num_valid_heads) {
    32	        return;
    33	    }
    34	
    35	    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    36	    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    37	    const int my_num_splits = end_split_idx - start_split_idx;
    38	    if (my_num_splits <= 1) {
    39	        return;
    40	    }
    41	    
    42	    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
    43	    
    44	    Tensor gLseAccum = make_tensor(
    45	        make_gmem_ptr((float*)params.lse_accum + start_split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + h_block_idx*BLOCK_SIZE_M),
    46	        Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
    47	        make_stride(params.stride_lse_accum_split, _1{})
    48	    );
    49	    Tensor gLse = make_tensor(
    50	        make_gmem_ptr((float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + h_block_idx*BLOCK_SIZE_M),
    51	        Shape<Int<BLOCK_SIZE_M>>{},
    52	        Stride<_1>{}
    53	    );
    54	    
    55	    __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];
    56	
    57	    // Wait for the previous kernel (the MLA kernel) to finish
    58	    cudaGridDependencySynchronize();
    59	
    60	    // Prefetch
    61	    static_assert(HEAD_DIM_V % (32*4) == 0);
    62	    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32*4);
    63	    float* oaccum_ptr = params.o_accum + start_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + (h_block_idx*BLOCK_SIZE_M + warp_idx)*params.stride_o_accum_h_q;
    64	    float4 datas[ELEMS_PER_THREAD];
    65	    CUTLASS_PRAGMA_UNROLL
    66	    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    67	        datas[i] = *(float4*)(oaccum_ptr + lane_idx*4 + i*128); // NOTE We don't use __ldg here since it is incompatible with PDL
    68	    }
    69	
    70	    // Warp #i gathers LseAccum for seq #i
    71	    {
    72	        constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);
    73	        float local_lse[NUM_LSE_PER_THREAD];
    74	        CUTLASS_PRAGMA_UNROLL
    75	        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
    76	            const int split_idx = i*32 + lane_idx;
    77	            local_lse[i] = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
    78	        }
    79	
    80	        float max_lse = -INFINITY;
    81	        CUTLASS_PRAGMA_UNROLL
    82	        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
    83	            max_lse = max(max_lse, local_lse[i]);
    84	        CUTLASS_PRAGMA_UNROLL
    85	        for (int offset = 16; offset >= 1; offset /= 2)
    86	            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    87	        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf
    88	
    89	        float sum_lse = 0;
    90	        CUTLASS_PRAGMA_UNROLL
    91	        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
    92	            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
    93	        CUTLASS_PRAGMA_UNROLL
    94	        for (int offset = 16; offset >= 1; offset /= 2)
    95	            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);
    96	
    97	        float global_lse = (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
    98	        if (lane_idx == 0)
    99	            gLse(warp_idx) = global_lse / (float)M_LOG2E;
   100	        
   101	        if (params.attn_sink != nullptr) {
   102	            int q_head_idx = h_block_idx*BLOCK_SIZE_M + warp_idx;
   103	            float attn_sink = __ldg(params.attn_sink + q_head_idx);
   104	            if (global_lse != INFINITY) {
   105	                // If attn_sink is +inf, global_lse will be +inf and scale factors will be exp2f(local_lse - inf) = 0 (since local_lse never becomes +inf)
   106	                // If attn_sink is -inf, this has no effect on global_lse
   107	                global_lse += log2f(1 + exp2f(attn_sink*CUDART_L2E_F - global_lse));
   108	            } else {
   109	                // We have no tokens to attend, so global lse should be attn_sink*CUDART_L2E_F (+inf if it's -inf or +inf)
   110	                global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink*CUDART_L2E_F;
   160	    }
   161	}
   162	
   163	
   164	#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)       \
   165	    [&] {                                                  \
   166	        if (NUM_SPLITS <= 32) {                            \
   167	            constexpr static int NAME = 32;                \
   168	            return __VA_ARGS__();                          \
   169	        } else if (NUM_SPLITS <= 64) {                     \
   170	            constexpr static int NAME = 64;                \
   171	            return __VA_ARGS__();                          \
   172	        } else if (NUM_SPLITS <= 96) {                     \
   173	            constexpr static int NAME = 96;                \
   174	            return __VA_ARGS__();                          \
   175	        } else if (NUM_SPLITS <= 128) {                    \
   176	            constexpr static int NAME = 128;               \
   177	            return __VA_ARGS__();                          \
   178	        } else if (NUM_SPLITS <= 160) {                    \
   179	            constexpr static int NAME = 160;               \
   180	            return __VA_ARGS__();                          \
   181	        } else {                                           \
   182	            FLASH_ASSERT(false);                           \
   183	        }                                                  \
   184	    }()
   185	
   186	
   187	template<typename ElementT>
   188	void run_flash_mla_combine_kernel(CombineParams &params) {
   189	    static constexpr int HEAD_DIM_V = 512;  // Since only this head dimension is supported by Flash MLA
   190	    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
   191	    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
   192	        constexpr int BLOCK_SIZE_M = 8;
   193	        constexpr int NUM_THREADS = BLOCK_SIZE_M*32;
   194	        constexpr size_t smem_size = BLOCK_SIZE_M*(NUM_SPLITS+1)*sizeof(float);
   195	        auto combine_kernel = &flash_fwd_mla_combine_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS>;
   196	        CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
   197	        // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
   198	        cudaLaunchAttribute attribute[1];
   199	        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
   200	        attribute[0].val.programmaticStreamSerializationAllowed = 1;
```
FlashMLA planner for reference (csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu):
```cpp
    60	
    61	    if (threadIdx.x == 0) {
    62	        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;
    63	
    64	        int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
    65	        num_splits_shared[0] = 0;
    66	        for (int i = 0; i < num_sm_parts; ++i) {
    67	            DecodingSchedMeta cur_meta;
    68	            cur_meta.begin_req_idx = now_req_idx;
    69	            cur_meta.begin_block_idx = now_block + first_block_idx_shared[now_req_idx];
    70	            cur_meta.begin_split_idx = now_n_split_idx;
    71	            cur_meta.is_first_req_splitted = (now_block != 0);
    72	            int remain_payload = payload;
    73	            while (now_req_idx < batch_size) {
    74	                int num_blocks = num_blocks_shared[now_req_idx];
    75	                int now_remain_blocks = num_blocks - now_block;
    76	                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
    77	                    cum_num_splits += now_n_split_idx + 1;
    78	                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
    79	                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
    80	                    ++now_req_idx;
    81	                    now_block = 0;
    82	                    now_n_split_idx = 0;
    83	                } else {
    84	                    if (remain_payload - fixed_overhead_num_blocks > 0) {
    85	                        now_block += remain_payload - fixed_overhead_num_blocks;
    86	                        ++now_n_split_idx;
    87	                        remain_payload = 0;
    88	                    }
    89	                    break;
    90	                }
    91	            }
    92	            cur_meta.end_req_idx = now_block > 0 ? now_req_idx : now_req_idx - 1;
    93	            cur_meta.end_block_idx = now_block > 0 ? now_block + first_block_idx_shared[now_req_idx] : (seqlens_k_shared[now_req_idx-1] == 0 ? 0 : last_block_idx_shared[now_req_idx-1] + 1);
    94	            cur_meta.is_last_req_splitted = cur_meta.end_block_idx != last_block_idx_shared[cur_meta.end_req_idx] + 1 && seqlens_k_shared[cur_meta.end_req_idx] != 0;
    95	            if (cur_meta.begin_req_idx == cur_meta.end_req_idx) {
    96	                cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
    97	            }
    98	            tile_scheduler_metadata_ptr[i] = cur_meta;
    99	        }
   100	        FLASH_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
   101	    }
   102	    __syncwarp();
   103	
   104	    for (int i = threadIdx.x; i <= batch_size; i += 32) {
   105	        num_splits_ptr[i] = num_splits_shared[i];
   106	    }
   107	}
   108	
   109	void run_get_decoding_sched_meta_kernel(GetDecodeSchedMetaParams &params) {
   110	    int smem_size = sizeof(int) * (params.b*5+1);
   111	    CHECK_CUDA(cudaFuncSetAttribute(get_mla_metadata_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
   112	    get_mla_metadata_kernel<<<1, 32, smem_size, params.stream>>>(params);
   113	    CHECK_CUDA_KERNEL_LAUNCH();
   114	}
   115	
   116	}
```

## vLLM diff (branch tp-sliced-moe)
```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index 6a46d7e048..4016361a04 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -281,6 +281,7 @@ if TYPE_CHECKING:
     VLLM_DCP_SPARSE_PREFILL: bool = True
     VLLM_DCP_ONE_SHOT_COLLECTIVES: bool = False
     VLLM_DCP_SPARSE_DECODE_WIDTH: int = 0
+    VLLM_DCP_DECODE_SCHED: bool = True
     VLLM_TIERED_MOE_KV_POOL_SEQS: int = 0
     VLLM_DCP_ONE_SHOT_FUSED: bool = True
     VLLM_DCP_ONE_SHOT_PREFILL: bool = True
@@ -2013,6 +2014,13 @@ environment_variables: dict[str, Callable[[], Any]] = {
     "VLLM_DCP_SPARSE_DECODE_WIDTH": lambda: int(
         os.getenv("VLLM_DCP_SPARSE_DECODE_WIDTH", "0")
     ),
+    # DCP fp8 sparse MLA on sm90 (non-prefill path): build FlashMLA's split
+    # schedule on device from the valid counts, covering only the blocks up to
+    # the longest live row and splitting them evenly, instead of FlashMLA's
+    # planner (full width, unbalanced splits for a single request).
+    "VLLM_DCP_DECODE_SCHED": lambda: bool(
+        int(os.getenv("VLLM_DCP_DECODE_SCHED", "1"))
+    ),
     # Tiered MoE: the KV pool in full-length (max_model_len) sequences, shared
     # by all running requests; 0 provisions max_num_seqs of them. Fewer than
     # max_num_seqs gives c requests a smaller pool (and the hot experts its
@@ -2392,6 +2400,7 @@ def compile_factors() -> dict[str, object]:
         # memory check: the traced graph and inductor keys are unchanged.
         "VLLM_DCP_SPARSE_PREFILL",
         "VLLM_DCP_SPARSE_DECODE_WIDTH",
+        "VLLM_DCP_DECODE_SCHED",
         "VLLM_TIERED_MOE_KV_POOL_SEQS",
         "VLLM_DCP_ONE_SHOT_PREFILL",
         "VLLM_TIERED_MOE_PREFILL_KERNEL",
diff --git a/vllm/v1/attention/backends/mla/flashmla_sparse.py b/vllm/v1/attention/backends/mla/flashmla_sparse.py
index 5e52a34854..3d805bf766 100644
--- a/vllm/v1/attention/backends/mla/flashmla_sparse.py
+++ b/vllm/v1/attention/backends/mla/flashmla_sparse.py
@@ -30,6 +30,7 @@ from vllm.v1.attention.backend import (
 )
 from vllm.v1.attention.backends.mla import skip_kv_stage
 from vllm.v1.attention.backends.mla.sparse_utils import (
+    dcp_decode_sched_meta,
     triton_convert_req_index_to_global_index,
     triton_filter_and_convert_dcp_index,
 )
@@ -241,11 +242,14 @@ class FlashMLASparseMetadata(AttentionMetadata):
 
 
 # DCP-filtered indices per shared top-k buffer (data_ptr), as
-# (key, indices, valid_counts). Layers without their own indexer (index_topk_freq
-# > 1) reuse the buffer the last indexer layer wrote, so within one forward their
-# conversion equals that layer's: the indexer layer stores it, the others reuse
-# it when the key (token count, block table, rank) matches.
-_DCP_CONVERTED_INDICES: dict[int, tuple[tuple, torch.Tensor, torch.Tensor]] = {}
+# (key, indices, valid_counts, decode schedule or None). Layers without their own
+# indexer (index_topk_freq > 1) reuse the buffer the last indexer layer wrote, so
+# within one forward their conversion equals that layer's: the indexer layer
+# stores it, the others reuse it when the key (token count, block table, rank)
+# matches.
+_DCP_CONVERTED_INDICES: dict[
+    int, tuple[tuple, torch.Tensor, torch.Tensor, tuple | None]
+] = {}
 
 # VLLM_DCP_SPARSE_DECODE_WIDTH: per device, int32 [max live slots in a row,
 # rows that overflowed the width], accumulated by the index conversion kernel
@@ -889,9 +893,10 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
                 q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
             )
         valid_counts: torch.Tensor | None = None
+        sched: tuple[torch.Tensor, torch.Tensor] | None = None
         if self.dcp_world_size > 1:
-            topk_indices, valid_counts = self._dcp_converted_indices(
-                topk_indices, attn_metadata
+            topk_indices, valid_counts, sched = self._dcp_converted_indices(
+                topk_indices, attn_metadata, decode_sched=self._dcp_decode_sched(q)
             )
         else:
             topk_indices = triton_convert_req_index_to_global_index(
@@ -921,12 +926,28 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         )
         fp8_metadata = attn_metadata.fp8_extra_metadata
 
-        _attn_out, _lse = self._fp8_flash_mla_kernel(
-            q=q.unsqueeze(0),  # unsqueeze to add batch_dim: (T, H, D) -> (1, T, H, D)
-            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
-            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
-            kernel_metadata=fp8_metadata,
-        )
+        if sched is not None:
+            _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
+                q.unsqueeze(0),
+                kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
+                topk_indices.unsqueeze(0),
+                None,
+                None,
+                *sched,
+                None,
+                None,
+                None,
+                512,
+                self.softmax_scale,
+                None,
+            )
+        else:
+            _attn_out, _lse = self._fp8_flash_mla_kernel(
+                q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
+                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
+                topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
+                kernel_metadata=fp8_metadata,
+            )
 
         if stager is not None and not self._reuses_topk:
             # after the anchor's own attention, so the copy does not slow it
@@ -945,7 +966,7 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         topk_indices: torch.Tensor,
         attn_metadata: FlashMLASparseMetadata,
     ) -> tuple[torch.Tensor, torch.Tensor]:
-        indices, valid_counts = self._dcp_converted_indices(
+        indices, valid_counts, _ = self._dcp_converted_indices(
             topk_indices, attn_metadata, prefill=True
         )
         # the combine's registered buffer, when it has one: no staging copy
@@ -988,12 +1009,25 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
                 kv_cache, kv_c_normed, k_pe, slot_mapping, k_scale
             )
 
+    def _dcp_decode_sched(self, q: torch.Tensor) -> bool:
+        """Whether this decode call can run on dcp_decode_sched_meta's schedule
+        (FlashMLA sm90 fp8 sparse decode, no head padding)."""
+        return (
+            envs.VLLM_DCP_DECODE_SCHED
+            and current_platform.is_device_capability(90)
+            and q.shape[1] == self.fp8_decode_padded_heads
+            and hasattr(torch.ops._flashmla_C, "sparse_decode_fwd")
+        )
+
     def _dcp_converted_indices(
         self,
         topk_indices: torch.Tensor,
         attn_metadata: FlashMLASparseMetadata,
         prefill: bool = False,
-    ) -> tuple[torch.Tensor, torch.Tensor]:
+        decode_sched: bool = False,
+    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
+        """This rank's compacted indices and valid counts, plus, with
+        ``decode_sched``, the matching FlashMLA decode schedule."""
         buffer = self.topk_indices_buffer
         slot = buffer.data_ptr() if buffer is not None else None
         bt = attn_metadata.block_table
@@ -1010,7 +1044,10 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         )
         cached = _DCP_CONVERTED_INDICES.get(slot) if slot is not None else None
         if self._reuses_topk and cached is not None and cached[0] == key:
-            return cached[1], cached[2]
+            sched = cached[3]
+            if decode_sched and sched is None:
+                sched = self._decode_sched_meta(cached[1], cached[2])
+            return cached[1], cached[2], sched if decode_sched else None
         width = envs.VLLM_DCP_SPARSE_DECODE_WIDTH
         stats = None
         if width and not prefill:
@@ -1044,9 +1081,21 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
             out_width=min(width, topk_indices.shape[1]) if stats is not None else None,
             stats=stats,
         )
+        sched = None
+        if decode_sched and not prefill:
+            sched = self._decode_sched_meta(indices, valid_counts)
         if slot is not None and not self._reuses_topk:
-            _DCP_CONVERTED_INDICES[slot] = (key, indices, valid_counts)
-        return indices, valid_counts
+            _DCP_CONVERTED_INDICES[slot] = (key, indices, valid_counts, sched)
+        return indices, valid_counts, sched
+
+    def _decode_sched_meta(
+        self, indices: torch.Tensor, valid_counts: torch.Tensor
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        # FlashMLA sm90: num_sm_parts = num_sms / s_q / (h_q / 64)
+        sms = num_compute_units(indices.device.index)
+        heads = self.fp8_decode_padded_heads
+        num_parts = max(sms // indices.shape[0] // (heads // 64), 1)
+        return dcp_decode_sched_meta(valid_counts, indices.shape[1], num_parts)
 
     def _fp8_flash_mla_kernel(
         self,
diff --git a/vllm/v1/attention/backends/mla/sparse_utils.py b/vllm/v1/attention/backends/mla/sparse_utils.py
index 76aa28aac5..c737ffec14 100644
--- a/vllm/v1/attention/backends/mla/sparse_utils.py
+++ b/vllm/v1/attention/backends/mla/sparse_utils.py
@@ -424,3 +424,73 @@ def triton_filter_and_convert_dcp_index(
         assert valid_counts is not None
         return out, valid_counts
     return out
+
+
+@triton.jit
+def _dcp_decode_sched_kernel(
+    valid_counts_ptr,
+    meta_ptr,
+    num_splits_ptr,
+    num_tokens,
+    width,
+    num_parts,
+    BLOCK_T: tl.constexpr,
+    BLOCK_P: tl.constexpr,
+):
+    live = tl.full((), 0, tl.int32)
+    for start in range(0, num_tokens, BLOCK_T):
+        offs = start + tl.arange(0, BLOCK_T)
+        counts = tl.load(valid_counts_ptr + offs, mask=offs < num_tokens, other=0)
+        live = tl.maximum(live, tl.max(counts, axis=0))
+    live = tl.minimum(live, width)
+    num_blocks = tl.maximum((live + 63) // 64, 1)
+    longest = (num_blocks + num_parts - 1) // num_parts
+    splits = (num_blocks + longest - 1) // longest
+    part = tl.arange(0, BLOCK_P)
+    active = part < splits
+    is_split = (splits > 1).to(tl.int32)
+    row = meta_ptr + part * 8
+    mask = part < num_parts
+    # DecodingSchedMeta: begin/end request, begin/end block, first split index,
+    # first/last request split, pad. Parts past `splits` start at request 1 = b
+    # and exit at once.
+    tl.store(row + 0, tl.where(active, 0, 1), mask=mask)
+    tl.store(row + 1, tl.zeros_like(part), mask=mask)
+    tl.store(row + 2, tl.where(active, num_blocks * part // splits, 0), mask=mask)
+    tl.store(row + 3, tl.where(active, num_blocks * (part + 1) // splits, 0), mask=mask)
+    tl.store(row + 4, tl.where(active, part, 0), mask=mask)
+    tl.store(row + 5, tl.where(active, is_split, 0), mask=mask)
+    tl.store(row + 6, tl.where(active, is_split, 0), mask=mask)
+    tl.store(row + 7, tl.zeros_like(part), mask=mask)
+    tl.store(num_splits_ptr + tl.arange(0, 2), tl.where(tl.arange(0, 2) == 0, 0, splits))
+
+
+def dcp_decode_sched_meta(
+    valid_counts: torch.Tensor, width: int, num_parts: int
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """FlashMLA sm90 sparse decode schedule for one request of T query tokens
+    whose index rows hold ``valid_counts`` live slots as a prefix (DCP).
+
+    Covers only the blocks up to the longest live row (the rest are -1 in
+    every row), split as evenly as the ``num_parts`` SM parts allow: the
+    smallest longest split, in the fewest splits. FlashMLA's own planner
+    charges 5 blocks of overhead per split, which for one request leaves the
+    splits unbalanced (9 + 3 blocks for 12 blocks over 2 parts).
+
+    Returns (tile_scheduler_metadata [num_parts, 8], num_splits [2]), int32.
+    """
+    assert valid_counts.dtype == torch.int32 and valid_counts.dim() == 1
+    assert width > 0 and width % 64 == 0 and 0 < num_parts <= 1024
+    meta = torch.empty((num_parts, 8), dtype=torch.int32, device=valid_counts.device)
+    num_splits = torch.empty(2, dtype=torch.int32, device=valid_counts.device)
+    _dcp_decode_sched_kernel[(1,)](
+        valid_counts,
+        meta,
+        num_splits,
+        valid_counts.shape[0],
+        width,
+        num_parts,
+        BLOCK_T=256,
+        BLOCK_P=triton.next_power_of_2(num_parts),
+    )
+    return meta, num_splits
```

Surrounding vLLM code: the decode forward after the kernel call (flashmla_sparse.py):
```python
   880	        Used when use_mixed_batch is True.
   881	        """
   882	        # Convert per-request indices to global slots (decode) or workspace
   883	        # offsets (prefill). Under DCP, keep only this rank's KV shard: owned
   884	        # slots are compacted to a contiguous prefix and the tail stays -1,
   885	        # which the kernel skips; the LSE-based cross-rank combine restores
   886	        # the full softmax.
   887	        if (
   888	            attn_metadata.dcp_prefill_local_len > 0
   889	            and q.shape[0] == attn_metadata.num_actual_tokens
   890	            and not torch.cuda.is_current_stream_capturing()
   891	        ):
   892	            return self._forward_dcp_sparse_prefill(
   893	                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
   894	            )
   895	        valid_counts: torch.Tensor | None = None
   896	        sched: tuple[torch.Tensor, torch.Tensor] | None = None
   897	        if self.dcp_world_size > 1:
   898	            topk_indices, valid_counts, sched = self._dcp_converted_indices(
   899	                topk_indices, attn_metadata, decode_sched=self._dcp_decode_sched(q)
   900	            )
   901	        else:
   902	            topk_indices = triton_convert_req_index_to_global_index(
   903	                attn_metadata.req_id_per_token,
   904	                attn_metadata.block_table,
   905	                topk_indices,
   906	                BLOCK_SIZE=attn_metadata.block_size,
   907	                NUM_TOPK_TOKENS=topk_indices.shape[1],
   908	            )
   909	        stager = (
   910	            skip_kv_stage.get_stager(q.device)
   911	            if self._skip_kv
   912	            and attn_metadata.num_prefills == 0
   913	            and q.shape[0] <= skip_kv_stage.limits()[0]
   914	            else None
   915	        )
   916	        if (
   917	            stager is not None
   918	            and self._reuses_topk
   919	            and (staged := stager.staged(kv_c_and_k_pe_cache, q.shape[0]))
   920	        ):
   921	            kv_c_and_k_pe_cache, topk_indices = staged
   922	
   923	        assert attn_metadata.fp8_extra_metadata is not None
   924	        assert isinstance(
   925	            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
   926	        )
   927	        fp8_metadata = attn_metadata.fp8_extra_metadata
   928	
   929	        if sched is not None:
   930	            _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
   931	                q.unsqueeze(0),
   932	                kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
   933	                topk_indices.unsqueeze(0),
   934	                None,
   935	                None,
   936	                *sched,
   937	                None,
   938	                None,
   939	                None,
   940	                512,
   941	                self.softmax_scale,
   942	                None,
   943	            )
   944	        else:
   945	            _attn_out, _lse = self._fp8_flash_mla_kernel(
   946	                q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
   947	                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
   948	                topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
   949	                kernel_metadata=fp8_metadata,
   950	            )
   951	
   952	        if stager is not None and not self._reuses_topk:
   953	            # after the anchor's own attention, so the copy does not slow it
   954	            stager.stage(kv_c_and_k_pe_cache, topk_indices, attn_metadata.slot_mapping)
   955	
   956	        # Output is (1, T, H, D_v) -> (T, H, D_v); lse is (1, H, T) -> (T, H)
   957	        lse = _lse.squeeze(0).transpose(0, 1)
   958	        if valid_counts is not None and not self.dcp_combine_drops_inf_lse:
   959	            lse = mask_empty_dcp_lse(lse, valid_counts)
   960	        return _attn_out.squeeze(0), lse
   961	
   962	    def _forward_dcp_sparse_prefill(
```
`mask_empty_dcp_lse` and the triton combine's LSE handling (vllm/v1/attention/ops/common.py):
```python
def mask_empty_dcp_lse(lse: torch.Tensor, valid_counts: torch.Tensor) -> torch.Tensor:
    """Correct the LSE of query rows whose top-k misses this DCP rank's shard.

    The FlashMLA sparse kernel returns lse=+inf for an all-invalid index row;
    the cross-rank combine needs -inf (empty softmax) or it produces NaN.
    """
    return lse.masked_fill((valid_counts == 0).unsqueeze(1), float("-inf"))

    50	    # shape = [N]
    51	    lse_offsets = (
    52	        num_n_offsets * lses_stride_N
    53	        + batch_idx * lses_stride_B
    54	        + head_idx * lses_stride_H
    55	    )
    56	
    57	    # calc final lse
    58	    lse = tl.load(lses_ptr + lse_offsets)
    59	    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    60	    lse_max = tl.max(lse, axis=0)
    61	    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    62	    lse -= lse_max
    63	    if IS_BASE_E:
    64	        lse_exp = tl.exp(lse)
    65	        lse_acc = tl.sum(lse_exp, axis=0)
    66	        lse = tl.log(lse_acc)
    67	    else:
    68	        lse_exp = tl.exp2(lse)
    69	        lse_acc = tl.sum(lse_exp, axis=0)
    70	        lse = tl.log2(lse_acc)
    71	    lse += lse_max
    72	
    73	    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    74	    tl.store(vlse_ptr + lse_offsets, lse)
    75	
    76	    # shape = [D]
    77	    output_offsets = (
    78	        batch_idx * outputs_stride_B
    79	        + head_idx * outputs_stride_H
    80	        + d_offsets * outputs_stride_D
    81	    )
    82	
    83	    # correct output
    84	    lse_offset = (
    85	        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    86	    )
    87	    lse_tmp = tl.load(lses_ptr + lse_offset)
    88	    lse_finally = lse_tmp - lse
    89	    lse_finally = tl.where(
    90	        (lse_finally != lse_finally) | (lse_finally == float("inf")),
    91	        -float("inf"),
    92	        lse_finally,
    93	    )
    94	    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    95	    output = tl.load(outputs_ptr + output_offsets)
```

## Tests added (tests/v1/attention/test_flashmla_sparse_dcp.py, appended; helpers _pack_fp8_ds_mla_cache / _run_sparse_decode run the stock planner path via flash_mla_with_kvcache)
```python


def _sched_impl():
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

    impl = object.__new__(FlashMLASparseImpl)
    impl.fp8_decode_padded_heads = NUM_HEADS
    return impl


def _run_scheduled_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DCP decode path on dcp_decode_sched_meta's schedule; (out, lse)."""
    sched = _sched_impl()._decode_sched_meta(indices, valid_counts)
    out, lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
        q.unsqueeze(0),
        cache.unsqueeze(-2),
        indices.unsqueeze(0),
        None,
        None,
        *sched,
        None,
        None,
        None,
        HEAD_DIM_V,
        HEAD_DIM**-0.5,
        None,
    )
    return out.squeeze(0), lse.squeeze(0).transpose(0, 1)


def _reference_decode(
    q: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 sparse attention over the dequantized fp8_ds_mla cache."""
    flat = cache.view(-1, 656)
    nope = flat[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = flat[:, 512:528].contiguous().view(torch.float32)
    v = nope * scales.repeat_interleave(128, dim=1)
    k = torch.cat([v, flat[:, 528:].contiguous().view(torch.bfloat16).float()], 1)
    valid = indices >= 0
    rows = indices.clamp(min=0).long()
    scores = torch.einsum("thd,tkd->thk", q.float(), k[rows]) * HEAD_DIM**-0.5
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", probs, v[rows]), lse


def _check_sched_meta(meta, num_splits, counts, width, num_parts):
    """The schedule covers [0, nb) contiguously, nb the blocks of the longest
    live row, in the fewest splits of the smallest possible longest split."""
    meta, num_splits = meta.cpu(), num_splits.cpu()
    live = min(max(counts), width)
    nb = max((live + 63) // 64, 1)
    longest = -(-nb // num_parts)
    splits = int(num_splits[1])
    assert num_splits.tolist() == [0, splits]
    assert splits == -(-nb // longest) <= num_parts
    active, idle = meta[:splits], meta[splits:]
    assert (idle[:, 0] == 1).all(), "idle parts must start at request b = 1"
    assert (active[:, 0] == 0).all() and (active[:, 1] == 0).all()
    assert active[:, 4].tolist() == list(range(splits))
    assert active[0, 2] == 0 and active[-1, 3] == nb
    assert torch.equal(active[1:, 2], active[:-1, 3])
    lens = active[:, 3] - active[:, 2]
    assert lens.min() >= 1 and lens.max() == longest
    assert (active[:, 5:7] == int(splits > 1)).all()


@pytest.mark.parametrize("num_parts", [1, 2, 4, 5, 8, 16, 33, 66, 132])
def test_dcp_decode_sched_meta_covers_live_blocks_evenly(num_parts: int):
    """Schedule invariants for every count pattern the DCP converter can give:
    empty rows, counts at and around block boundaries, counts past the width
    (rows that overflowed it), and the longest row late in a long batch."""
    from vllm.v1.attention.backends.mla.sparse_utils import dcp_decode_sched_meta

    torch.manual_seed(17)
    for width in (64, 576, 768, 2048):
        patterns = [
            [0],
            [0] * 9,
            [1],
            [63, 0, 2],
            [64],
            [65, 3],
            [width - 1],
            [width],
            [width + 1, 0],
            [4 * width, 7],
            [0] * 299 + [width // 2 + 1],
            torch.randint(0, width + 1, (64,)).tolist(),
        ]
        for counts in patterns:
            vc = torch.tensor(counts, dtype=torch.int32, device="cuda")
            meta, num_splits = dcp_decode_sched_meta(vc, width, num_parts)
            assert meta.shape == (num_parts, 8) and num_splits.shape == (2,)
            _check_sched_meta(meta, num_splits, counts, width, num_parts)


def _dcp_rows(counts, width, num_rows_cache, device):
    """Compacted DCP index rows: counts[i] live cache rows (repeats allowed,
    the last cache row always among them) as a prefix, -1 after."""
    indices = torch.full((len(counts), width), -1, dtype=torch.int32, device=device)
    for i, c in enumerate(counts):
        n = min(c, width)
        if n:
            indices[i, :n] = torch.randint(0, num_rows_cache, (n,), device=device)
            indices[i, n - 1] = num_rows_cache - 1
    return indices, torch.tensor(counts, dtype=torch.int32, device=device)


def _counts(pattern: str, t: int, width: int) -> list[int]:
    g = torch.Generator().manual_seed(t * 7919 + width)
    if pattern == "live":  # serving-like: ~500-of-768 live, a few empty rows
        hi = min(width, 540)
        c = torch.randint(hi * 5 // 6, hi + 1, (t,), generator=g).tolist()
        return [0 if i % 5 == 3 else x for i, x in enumerate(c)]
    if pattern == "boundary":
        edges = [1, 63, 64, 65, 127, 128, 129, width - 1, width, 0]
        return [edges[(i * 3) % len(edges)] for i in range(t)]
    if pattern == "single_key":
        return [1 if i == t - 1 else 0 for i in range(t)]
    if pattern == "empty":
        return [0] * t
    if pattern == "overflow":  # counts past the width: rows truncated to it
        return [width + 1 + i if i % 2 == 0 else width // 3 for i in range(t)]
    raise ValueError(pattern)


@pytest.mark.parametrize(
    "pattern", ["live", "boundary", "single_key", "empty", "overflow"]
)
@pytest.mark.parametrize("width", [64, 768, 2048])
@pytest.mark.parametrize("num_query_tokens", [1, 2, 7, 16, 33, 64, 140])
def test_dcp_decode_sched_matches_planner_and_reference(
    num_query_tokens: int, width: int, pattern: str
):
    """The scheduled decode (early stop at the longest live row, balanced
    splits) attends like FlashMLA's own planner over the full width, up to
    fp32 split-boundary reordering, and like an fp32 reference. Empty rows keep
    a non-finite LSE (the combine drops it) and a finite output (it is
    weighted by zero, which a NaN would survive)."""
    torch.manual_seed(num_query_tokens * 131 + width)
    device = torch.device("cuda")
    t, num_ctx = num_query_tokens, 3000
    kv_c = 3.0 * torch.randn(num_ctx, 512, dtype=torch.bfloat16, device=device)
    kv_c[::7] = 0  # all-zero groups take the unit-scale path
    kv_c[1::11, :128] *= 200  # one group's scale far from the others'
    k_pe = 0.5 * torch.randn(num_ctx, 64, dtype=torch.bfloat16, device=device)
    cache = _pack_fp8_ds_mla_cache(kv_c, k_pe)
    q = 2.0 * torch.randn(t, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    counts = _counts(pattern, t, width)
    indices, valid_counts = _dcp_rows(counts, width, num_ctx, device)

    out, lse = _run_scheduled_decode(q, cache, indices, valid_counts)
    ref_out, ref_lse = _run_sparse_decode(q, cache, indices)
    torch.cuda.synchronize()

    empty = valid_counts == 0
    assert torch.isfinite(out).all() and torch.isfinite(ref_out).all()
    assert not torch.isfinite(lse[empty]).any()
    assert not torch.isfinite(ref_lse[empty]).any()
    assert torch.isfinite(lse[~empty]).all()
    torch.testing.assert_close(out[~empty], ref_out[~empty], atol=2e-3, rtol=1.6e-2)
    torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=1e-4, rtol=1e-5)

    fp32_out, fp32_lse = _reference_decode(q, cache, indices)
    torch.testing.assert_close(
        out[~empty].float(), fp32_out[~empty], atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(lse[~empty], fp32_lse[~empty], atol=2e-3, rtol=1e-4)


def test_dcp_decode_sched_replays_in_cuda_graph_with_changing_counts():
    """Schedule and attention captured once replay correctly as the live
    counts move between steps: grow from one key to the full width, shrink,
    go fully empty and come back."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    t, width, num_ctx = 32, 768, 4096
    cache = _pack_fp8_ds_mla_cache(
        torch.randn(num_ctx, 512, dtype=torch.bfloat16, device=device),
        0.5 * torch.randn(num_ctx, 64, dtype=torch.bfloat16, device=device),
    )
    q = torch.randn(t, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    indices = torch.full((t, width), -1, dtype=torch.int32, device=device)
    valid_counts = torch.zeros(t, dtype=torch.int32, device=device)

    def step():
        return _run_scheduled_decode(q, cache, indices, valid_counts)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = step()

    for counts in (
        [1] * t,
        [width] * t,
        [500 + i for i in range(t)],
        [0] * (t - 1) + [65],
        [0] * t,
        [width + 9] + [64] * (t - 1),
        [64] * t,
    ):
        new_indices, new_counts = _dcp_rows(counts, width, num_ctx, device)
        indices.copy_(new_indices)
        valid_counts.copy_(new_counts)
        graph.replay()
        ref_out, ref_lse = _run_sparse_decode(q, cache, indices)
        torch.cuda.synchronize()
        live = valid_counts > 0
        assert torch.isfinite(out).all() and not torch.isfinite(lse[~live]).any()
        torch.testing.assert_close(out[live], ref_out[live], atol=2e-3, rtol=1.6e-2)
        torch.testing.assert_close(lse[live], ref_lse[live], atol=1e-4, rtol=1e-5)


def test_dcp_reuse_layers_share_the_indexer_layers_schedule():
    """Layers that reuse the indexer layer's top-k get its converted indices
    and schedule without recomputing; a cached conversion without a schedule
    (computed with it off) gets one on demand, and prefill never gets one."""
    from types import SimpleNamespace

    from vllm.utils.platform_utils import num_compute_units
    from vllm.v1.attention.backends.mla import flashmla_sparse

    torch.manual_seed(23)
    device = torch.device("cuda")
    t, num_ctx, world = 16, 8192, 4
    buffer = torch.stack(
        [torch.randperm(num_ctx, device=device)[:TOPK] for _ in range(t)]
    ).to(torch.int32)
    meta = SimpleNamespace(
        block_table=torch.arange(
            num_ctx // world // BLOCK_SIZE, dtype=torch.int32, device=device
        ).unsqueeze(0),
        req_id_per_token=torch.zeros(t, dtype=torch.int32, device=device),
        block_size=BLOCK_SIZE,
        cp_kv_cache_interleave_size=1,
    )

    def layer(reuses: bool):
        impl = _sched_impl()
        impl.topk_indices_buffer = buffer
        impl.dcp_world_size, impl.dcp_rank = world, 1
        impl._reuses_topk = reuses
        return impl

    flashmla_sparse._DCP_CONVERTED_INDICES.clear()
    anchor, reuse = layer(False), layer(True)
    idx, counts, sched = anchor._dcp_converted_indices(buffer, meta, decode_sched=True)
    ridx, rcounts, rsched = reuse._dcp_converted_indices(
        buffer, meta, decode_sched=True
    )
    assert ridx is idx and rcounts is counts and rsched is sched
    num_parts = max(num_compute_units(device.index or 0) // t, 1)
    _check_sched_meta(*sched, counts.tolist(), TOPK, num_parts)
    assert reuse._dcp_converted_indices(buffer, meta)[2] is None

    flashmla_sparse._DCP_CONVERTED_INDICES.clear()
    anchor._dcp_converted_indices(buffer, meta)
    _, _, lazy = reuse._dcp_converted_indices(buffer, meta, decode_sched=True)
    assert all(torch.equal(a, b) for a, b in zip(lazy, sched))

    _, _, none = anchor._dcp_converted_indices(
        buffer, meta, prefill=True, decode_sched=True
    )
    assert none is None
    flashmla_sparse._DCP_CONVERTED_INDICES.clear()
```

Also: agent_space/.../regs_hashes.py hashes out+lse for every T in 1..64 plus 100, 140, widths 64/576/768/2048, patterns live/boundary/single/empty/overflow/full, under the planner and under the balanced schedule (Python transliteration of the Triton rule), once per library (prod .so, stock rebuild, register-split build) in separate processes; then compares prod vs stock and prod vs regs for bitwise equality.

## Questions
1. Any input the DCP path can produce for which the schedule skips a block containing a live key, or emits an invalid DecodingSchedMeta (empty split, split count > num_sm_parts / combine MAX_SPLITS, wrong begin_split_idx, idle part not idle)? Consider padded CUDA-graph rows, T > 132, T not dividing 132, h_q = 128 padded heads, num_compute_units vs FlashMLA's Arch().num_sms, valid_counts longer than the index rows, etc.
2. CUDA-graph / stream hazards: the schedule tensors are allocated per call inside capture; reuse layers (no indexer of their own) get the indexer layer's cached (indices, valid_counts, sched) via `_DCP_CONVERTED_INDICES`; the indexer itself may run on a side stream. Lifetime/ordering problems?
3. The 1-row-of-splits path (splits == 1 -> is_split = 0, kernel writes out/lse directly; combine skipped for my_num_splits <= 1). Any difference in empty-row semantics vs the planner path (which almost always splits)? E.g. lse=+inf vs what the combine returns; out values for empty rows.
4. The setmaxnreg change: is 208/160/136 safe (sum 504 = 3x168)? Is the host-side check sound (cudaFuncGetAttributes numRegs is the launch-time per-thread count)? Anything else that changes with the producer at 136 regs (spills? none in -Xptxas -v), or WG1 dealloc semantics?
5. Test gaps: what adversarial cases are missing or what assertion is too weak (tolerances, comparing against the planner, the fp32 reference)? Anything in the tests that would pass while the feature is broken?
6. Anything else you would block shipping on.
