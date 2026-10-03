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
  spec_config="{\"method\":\"dflash\",\"model\":\"${drafter}\",\"num_speculative_tokens\":${SPEC_K},\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\",\"draft_sample_method\":\"greedy\"}"
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
# 9 (was 7): long-context 8192-token chunks take the DSA sparse path, whose
# FlashMLA kernel allocates 2 GiB the startup profile (dense path) never sees.
# At 7 the server had 5.4 GiB free and OOMed on the first 60K-token prompt,
# with or without the prefill MoE kernel; at 9 it has 7.2 GiB free and runs
# (stress_long.sh, 2026-10-03). Costs ~3% of hot experts.
export TIERED_MOE_HBM_RESERVE_GB="${RESERVE_GB:-9}"
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
  --gpu-memory-utilization 0.90 \
  --max-model-len "${MAX_MODEL_LEN:-400000}" \
  $([[ -n "${PREFIX_CACHING-1}" ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching) \
  --generation-config vllm \
  --served-model-name glm53-w4a16-tiered \
  "${extra[@]}" ${SERVE_EXTRA:-}
