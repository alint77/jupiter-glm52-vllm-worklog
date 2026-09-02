#!/usr/bin/env bash
# Fresh sglang tree for the DFlash2/MTP acceptance replication (2026-09-02).
# Deliberately does NOT source the old venv: that one editable-installs the
# GLM-5.2 fork and needs PYTHONPATH shadowing. This venv has exactly one tree.
module load Stages/2026
module load GCC/14.3.0 CUDA/13 CMake/3.31.8 NCCL/default-CUDA-13

export SGLANG_ROOT=/e/fscratch/profound/naeimitabiei1/sglang-fresh-20260902
export SGLANG_VENV="$SGLANG_ROOT/venv"
export GLM53_NVFP4=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-NVFP4
export DFLASH2_DRAFT=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-DFlash2

# deep_gemm asserts on CUDA_HOME at import; without it every sglang import dies.
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

export UV_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/uv-cache
export XDG_CACHE_HOME=/e/fscratch/profound/naeimitabiei1/cache
export SGLANG_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/sglang-cache-fresh
# SGLANG_CACHE_DIR does NOT cover the sglang.kernels JIT cache: its resolver
# hardcodes the fallback (kernels/jit/utils/compile/cache.py:302,
# `envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"`) instead of
# deriving it from SGLANG_CACHE_DIR the way SGLANG_DG_CACHE_DIR does. Verified:
# 5.9 GB had accumulated under $HOME/.cache/sglang. Home is quota-limited and
# slow, so pin it explicitly.
export SGLANG_JIT_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/sglang-cache-fresh/jit
export TILELANG_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/tilelang-cache
export TORCH_EXTENSIONS_DIR=/e/fscratch/profound/naeimitabiei1/torch-extensions
export FLASHINFER_WORKSPACE_BASE=/e/fscratch/profound/naeimitabiei1/flashinfer
mkdir -p "$SGLANG_CACHE_DIR" "$TILELANG_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"

# No aarch64 FA3 build: a short-KV prefill otherwise takes the dense MHA
# one-shot path and dies on `from ... import flash_ops`.
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0

# --- Internode comms (JUPITER Booster fabric) ---------------------------
# Hardware, per the JSC configuration page: each node is 4x GH200, and each
# GH200 has its OWN ConnectX-7 NDR200 HCA (4x200 Gbit/s = 800 Gbit/s/node),
# on a Dragonfly+ topology with adaptive routing. Intra-node GPU-GPU is
# NVLink between *pairs* (300 GB/s/pair), not a full NVSwitch mesh.
#
# The JSC "Tuning for Large-Scale Execution" page gives UCX/MPI settings
# (UCX_TLS, UCX_RNDV_THRESH); those govern OpenMPI/ParaStationMPI and do not
# apply here -- sglang's TP transport is NCCL. These are the NCCL analogues
# for this fabric:
# Make every run state its own transport instead of leaving it inferable:
# the first 2-node runs logged nothing, and confirming IB (rather than an
# Ethernet fallback) needed forensics on /proc/<pid>/fd for uverbs handles.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

export NCCL_CROSS_NIC=1            # Dragonfly+ is not rail-optimised end to
                                   # end; let a ring leave via a different NIC
                                   # than it entered, which adaptive routing
                                   # then spreads.
export NCCL_IB_QPS_PER_CONNECTION=4  # more QPs per connection gives adaptive
                                     # routing more paths to balance over.
export NCCL_NET_GDR_LEVEL=SYS      # GPUDirect RDMA: the HCA is local to its
                                   # own superchip, so keep GPU<->NIC direct.
export NCCL_IB_TIMEOUT=22          # the JSC FAQ documents "Transport retry
export NCCL_IB_RETRY_CNT=10        # count exceeded" failures on this fabric.
# NCCL_IB_HCA is deliberately unset: one HCA per GPU means NCCL's own
# topology detection already pairs each rank with its local rail, and pinning
# it by name would break if device ordering differs between nodes.
#
# NOTE: every setting here moves THROUGHPUT only. Acceptance length is a
# property of the model math and is invariant to the transport, so the
# acceptance arms are valid with or without this block.

# FA3 loading order: sglang first tries the HF kernels hub repo
# (kernels-community/sgl-flash-attn3), which publishes x86_64 only, and falls
# back to sgl_kernel.flash_ops only when this is set. So a locally built FA3
# is unreachable without it (flash_attention_v3.py:_load_fa3_kernels).
export SGLANG_USE_SGL_FA3_KERNEL=1

# Compute nodes have no external network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

source "$SGLANG_VENV/bin/activate"

sglang_tree_check() {
  local got
  got=$(python -c "import sglang,os;print(os.path.dirname(sglang.__file__))" 2>/dev/null)
  case "$got" in
    "$SGLANG_VENV"/*) echo "sglang tree OK: $got" ;;
    *) echo "*** WRONG SGLANG TREE: $got (want under $SGLANG_VENV) ***" >&2; return 1 ;;
  esac
}
