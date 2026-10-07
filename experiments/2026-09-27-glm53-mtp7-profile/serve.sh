#!/usr/bin/env bash
# GLM-5.3 W4A16, c=1, 400K context. Defaults are the settled baseline: the
# DFlash2 drafter (7 tokens, eager), DCP4, exact replicas, the INT4 one-kernel
# decode MoE, reserve 7, capture [8] (SPEC=mtp / DCP=1 / REPLICAS= /
# VLLM_TIERED_MOE_DECODE_KERNEL=0 undo them; SPEC=mtp captures [1, 8]).
# Started from the spec-comparison
# mtp7 arm (../2026-09-04-spec-comparison/arm.sh) plus the production
# launcher's cold prefetch, with
#   * prefix caching on (PREFIX_CACHING= turns it off, so a repeated prompt
#     really prefills)
#   * --profiler-config when TRACE_ROOT is set (TRACE_STACK=true adds Python
#     stacks; with CUDAGRAPH_MODE=NONE each kernel maps back to its source)
# Both drafters verify 8 tokens per step, the same shape as MiMo's DFlash K=7.
# Run on the node from the repo root with the environment loaded.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_SERVER_DEV_MODE=1
c=/e/fscratch/profound/${USER}/caches
# SERVE_CACHE_ROOT: a separate compile cache, for servers running at once on
# several nodes (concurrent writers corrupt shared torch.compile artifacts)
export VLLM_CACHE_ROOT=${SERVE_CACHE_ROOT:-${c}/marlin/vllm-cache-glm53-mtp7}
export TRTLLM_DG_CACHE_DIR=${c}/marlin/trtllm-dg-glm53-mtp7
export TRITON_CACHE_DIR=${c}/triton
export TORCHINDUCTOR_CACHE_DIR=${c}/inductor
export FLASHINFER_CACHE_DIR=${c}/flashinfer
export TIERED_MOE_MODEL_PATH=/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16
# agentic-3239-r2000: built from the MiMo-workload capture, all 3,239 hot per GPU
# frequency-ranked, up to 2,000 replicas; -0.94 +- 0.12 ms/step against
# glm53-w4a16-2496.json (ab-gA, ab-gB)
export TIERED_MOE_PLACEMENT_PROFILE=${PWD}/agent_space/profiles/${PROFILE:-glm53-w4a16-agentic-3239-r2000.json}
# SPEC=dflash2: the DFlash2 drafter (7 tokens in one pass) instead of MTP7. The
# tiered planner now budgets its KV, so it runs at MTP's 7 GB reserve (free HBM
# stayed flat at 6.5 GiB over the task set; 10 GB cost ~150 hot experts and
# ~0.8 ms/step); it has no size-1 draft passes to capture; and compile_sizes
# stays empty as in production
# (../2026-09-04-glm53-c1-df2/server.sbatch: Triton autotune runs out of shared
# memory, and a compiled selector breaks the draft's captured region).
SPEC="${SPEC:-dflash2}"
# SPEC_K draft tokens per step; the verify step is SPEC_K + 1 tokens.
SPEC_K="${SPEC_K:-7}"
verify=$((SPEC_K + 1))
if [[ "${SPEC}" == dflash2 ]]; then
  drafter=/e/fscratch/profound/${USER}/models/GLM-5.3-DFlash2
  # Drafter KV and weights in fp8 (2026-10-06-skip-layer-kv-grace): acceptance
  # unchanged on the agentic set; frees 1.15 GiB (KV) + 0.64 GB (weights) per
  # rank for hot experts. DRAFT_KV_DTYPE=auto / DRAFT_QUANT= restore bf16.
  dq=""; [[ -n "${DRAFT_QUANT-fp8_per_channel}" ]] && dq=",\"quantization\":\"${DRAFT_QUANT-fp8_per_channel}\""
  spec_config="{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":${SPEC_K},\"kv_cache_dtype\":\"${DRAFT_KV_DTYPE:-fp8}\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"${dq}}"
  : "${CAPTURE_SIZES:=${verify},16,32,64,128,256,384,512,640,768,896,1024}" "${COMPILE_SIZES:=}"
elif [[ "${SPEC}" == dspark-* ]]; then
  # SPEC=dspark-redhat | dspark-alaya: GLM-5.3 DSpark drafters (block size 8,
  # so vLLM requires SPEC_K >= 8; the verify step is then 9 tokens, past the
  # one-kernel MoE's 8, and takes the two-tier Marlin path).
  drafter=/e/fscratch/profound/${USER}/models/GLM-5.3-DSpark-${SPEC#dspark-}
  [[ "${SPEC}" == dspark-redhat ]] && drafter=/e/fscratch/profound/${USER}/models/GLM-5.3-DSpark-RedHatAI
  [[ "${SPEC}" == dspark-alaya ]] && drafter=/e/fscratch/profound/${USER}/models/GLM-5.3-DSpark-AlayaNeW
  SPEC_K="${SPEC_K_DSPARK:-8}"; verify=$((SPEC_K + 1))
  spec_config="{\"method\":\"dspark\",\"model\":\"${drafter}\",\"num_speculative_tokens\":${SPEC_K},\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}"
  : "${CAPTURE_SIZES:=${verify}}" "${COMPILE_SIZES:=}"
else
  spec_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K}}"
  : "${CAPTURE_SIZES:=1,${verify}}" "${COMPILE_SIZES:=${verify}}"
fi
# Reserve 7 with 4096-token prefill chunks and the prefill-kernel-aware planner
# (vllm b41117c34d): 3244 hot experts per rank (3033 at 8192 / reserve 9 under
# the old budget), 3.25 GiB free after startup (2.53 required), a 390K-token
# session with 40K uncached jumps ran clean (plan4k, 2026-10-03). Reserve 6
# fails the startup check (2.55 GB free, 2.71 required).
# The planner now charges a DFlash drafter's weights (1.861 GB per rank in bf16,
# 1.220 GB with DRAFT_QUANT=fp8_per_channel; vllm 2026-10-07) that reserve 7
# used to absorb, so DFlash2 runs at the same margin with 7 - 1.861.
# DFlash2 with the full memory stack, swept 2026-10-07 (2026-10-07-reserve-sweep,
# stress_long.sh to 388K, 40K uncached jumps): 5.14 / 4.9 / 4.7 / 4.55 all run
# clean (3510 / 3522 / 3531 / 3538 hot per rank); 4.4 refuses to start. 4.7
# keeps 0.15 GiB over the startup check (4.55 had one rank at +0.01).
# vllm f3c882485e..794eda5446 (DCP prefill workspaces, one shared NCCL
# communicator, embedding on Grace, reserve floor 0.5) free ~3.5 GiB; swept
# again (2026-10-07-mem-reclaim): 2.0 / 1.6 run 388K clean, 1.3 and below
# refuse to start. 1.7: 3676 hot per rank (+145), ~0.17 GiB over the check;
# -0.22 ms/step agentic, -0.27 at 50-130K, GSM8K unchanged.
reserve_default=7
[[ "${SPEC}" == dflash2 ]] && reserve_default=1.7
export TIERED_MOE_HBM_RESERVE_GB="${RESERVE_GB:-${reserve_default}}"
# Cold prefetch: two slots (vLLM default), so layer L+1's copy runs under layer
# L's MoE; from 512 tokens (~1.3 ms copy vs ~2 ms of layer compute).
# Inside captured prefill graphs too (vllm a3e5a9c5f1): median TTFT at 512/768/
# 1024 tokens 176/236/240 -> 135/172/205 ms, same node (2026-10-02).
export VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS="${VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS:-512}"
# Prefill CUDA graphs (piecewise, up to 1024 tokens) need ~0.5 GB of graph pool
# the planner does not budget; it comes out of the free-HBM margin (5.5 GB).
# Median TTFT at 512 new tokens 188-222 -> 174 ms, decode unchanged (2026-10-01).
export VLLM_TIERED_MOE_OBSERVED_HBM_TOLERANCE_GB="${VLLM_TIERED_MOE_OBSERVED_HBM_TOLERANCE_GB:-1.5}"
# one-kernel INT4 decode MoE, replicas balanced by time in the kernel
export VLLM_TIERED_MOE_DECODE_KERNEL="${VLLM_TIERED_MOE_DECODE_KERNEL:-1}"
# wgmma prefill MoE for steps of > 8 tokens: median TTFT 512/768/1024/2048/4096
# 134/171/202/327/990 -> 122/135/154/243/873 ms same node, GSM8K 91.25 -> 90.0%
# (within noise) (vllm 23a50e73b2, 2026-10-03)
export VLLM_TIERED_MOE_PREFILL_KERNEL="${VLLM_TIERED_MOE_PREFILL_KERNEL:-1}"
# DCP's small gathers / reduce-scatters as one-shot kernels on the custom
# all-reduce buffers instead of NCCL: -3.07 +- 0.20 ms/step (ab-oC, ab-oD)
export VLLM_DCP_ONE_SHOT_COLLECTIVES="${VLLM_DCP_ONE_SHOT_COLLECTIVES:-1}"
# [1, 8]: 8 is the verify step; 1 is MTP's draft-decode passes (positions
# 1..K-1, one token each at c=1). Without a size-1 entry the speculator's
# decode graph is silently skipped and those passes run eagerly.
# FUSE_AR_RMS (default on): all-reduce + residual + RMSNorm fused (FlashInfer
# trtllm; auto picks it on these multicast-less nodes): -0.44 ms/step (95% CI
# -0.59 .. -0.28), acceptance unchanged, same-node pairs on 4 nodes (2026-09-30).
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES:-1,8}],\"compile_sizes\":[${COMPILE_SIZES-8}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":${FUSE_AR_RMS:-true}}}"
extra=()
# KV placement (2026-10-06-skip-layer-kv-grace): the 57 index-share layers' MLA
# KV on Grace, staged into HBM per decode step (MLA_CACHE_TIER=hbm restores it),
# and the drafter's KV on Grace (VLLM_TIERED_MOE_DRAFT_KV_HOST=0 restores it).
extra+=(--mla-cache-tier "${MLA_CACHE_TIER:-skip_host_uva}")
export VLLM_TIERED_MOE_DRAFT_KV_HOST="${VLLM_TIERED_MOE_DRAFT_KV_HOST:-1}"
# Input embedding table on Grace (a step gathers 8 rows; -0.44 GiB HBM);
# VLLM_TIERED_MOE_EMBED_HOST=0 restores it. Reserve 1.7 assumes it.
export VLLM_TIERED_MOE_EMBED_HOST="${VLLM_TIERED_MOE_EMBED_HOST:-1}"
# bf16 linears at <= 8 tokens on the weight-streaming kernel (vllm 86f871f450,
# 2026-10-07-skinny-gemm-v2): -0.47 +- 0.03 ms/step agentic, -0.50 at
# 50-130K, GSM8K unchanged. VLLM_DECODE_GEMM=0 restores cuBLAS.
export VLLM_DECODE_GEMM="${VLLM_DECODE_GEMM:-1}"
# REPLICAS=exact activates the profile's Grace replicas (1,351-1,962 per rank
# in the agentic profile). Replicas add pinned Grace the planner does not see
# each worker's own share of, hence the larger host reserve (as for MiMo).
REPLICAS="${REPLICAS-exact}"
if [[ -n "${REPLICAS}" ]]; then
  extra+=(--tiered-moe-replica-assignment "${REPLICAS}"
          --tiered-moe-host-reserve-gb "${HOST_RESERVE_GB:-16}")
fi
# NSYS_OUT=<report path>: Nsight Systems over the profile windows only
# (cudaProfilerApi range), CUDA graphs as single ranges, so the host overhead
# is seen without the torch profiler's per-kernel CUPTI cost.
if [[ -n "${NSYS_OUT:-}" ]]; then
  export SERVER_WRAPPER="${NSYS_BIN:-/e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys} profile -o ${NSYS_OUT} --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true --sample=none --cpuctxsw=none"
  extra+=(--profiler-config "{\"profiler\":\"cuda\"}")
elif [[ -n "${TRACE_ROOT:-}" ]]; then
  mkdir -p "${TRACE_ROOT}"
  extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TRACE_ROOT}\",\"torch_profiler_with_stack\":${TRACE_STACK:-false},\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}")
fi
exec agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config "${spec_config}" \
  --decode-context-parallel-size "${DCP:-4}" \
  --dcp-comm-backend "${DCP_COMM:-ag_rs}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len "${MAX_MODEL_LEN:-400000}" \
  $([[ -n "${PREFIX_CACHING-1}" ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching) \
  --generation-config vllm \
  --served-model-name glm53-w4a16-tiered \
  "${extra[@]}" ${SERVE_EXTRA:-}
