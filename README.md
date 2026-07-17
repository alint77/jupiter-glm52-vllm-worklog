# JUPITER GLM-5.2 vLLM worklog

Experimental worklog for serving the 361.06 GiB
`lowbitcoffee/GLM-5.2-W4A16` checkpoint on one JUPITER Booster node. The goal
is to improve batch-one decode at up to 400K context by keeping hot MoE experts
in HBM and executing colder experts from coherent Grace memory through CUDA
UVA. The current 37.5 tok/s result is the baseline; the project target starts
at 100 tok/s.

## Platform

- One Booster node with four NVIDIA GH200 Superchips
- Four 96 GiB Hopper HBM devices connected by NVLink
- Four 72-core Grace CPUs: 288 Arm cores total
- About 857 GiB of NUMA-visible Grace + HBM memory
- ExaSTORE/GPFS project storage; Booster nodes have no external internet
- vLLM commit `d08eebad162bbd1f2e99cca550313daaa81c7654`
- PyTorch 2.11, CUDA 13, TP=4, EP=4

Each rank owns 64 of the model's 256 routed experts. The baseline offloads
40.33 GiB of expert parameters per rank through vLLM's existing UVA offloader,
uses `fp8_ds_mla` KV cache, Inductor mode 3, and full/piecewise CUDA graphs.

## Baseline

Batch one, random input, deterministic decoding, 256 output tokens:

| Input | TTFT | Approx. prefill | Decode | Total |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.990 s | 4,139 tok/s | 37.06 tok/s | 7.87 s |
| 32,768 | 7.652 s | 4,282 tok/s | 37.06 tok/s | 14.53 s |
| 399,744 | 120.853 s | 3,308 tok/s | 37.57 tok/s | 127.64 s |

At idle, the working configuration uses about 90.7 GiB HBM per GPU, leaves
6.58 GiB free, and provides a 574,336-token KV capacity. Detailed settings and
result links are in [the baseline report](baseline/baseline-summary.md).

## How we got here

1. Downloaded and checksum-pinned the eight-shard model on the internet-facing
   login node, then used the local immutable checkpoint offline on Booster.
2. Built an editable vLLM environment with `uv` after loading the JUPITER 2026,
   GCC 14.3, CUDA 13, CMake, NCCL, ccache, and Ninja modules.
3. Enabled TP4/EP4, per-rank expert filtering, NUMA binding, 40 GiB/rank UVA
   expert offload, FP8 MLA KV cache, Inductor, autotuning, and CUDA graphs.
4. Moved compiler caches from the quota-limited home directory to scratch.
5. Disabled `fuse_allreduce_rms`; that fused warmup path produced an illegal
   memory access on this stack.
6. Reduced `gpu-memory-utilization` from 0.94 to 0.90. The former left only
   about 2.95 GiB free and failed at an 8K chunk during 32K prefill; 0.90 leaves
   enough transient HBM while preserving more than 400K KV capacity.
7. Repeated warmed 4K and 32K measurements, then completed the 399,744 + 256
   full-context case.

ExaFlash staging was investigated but not used; these results load directly
from ExaSTORE.

## Implementation status

Phase 0 development began on 2026-07-17 on branch
`tiered-moe-grace-view`. The first slice adds a capability-gated CUDA alias for
ordinary pageable Grace allocations, without pinning, registration, or a copy.
Correctness tests cover address identity, bidirectional visibility, ownership,
and invalid storage. Runtime qualification passes for PyTorch CUDA kernels, but
the driver migrates the pages from Grace node 0 to GPU-HBM node 4. Preferred-host
advice, read-mostly advice, and `mlock` did not preserve the physical LPDDR tier.
The direct pageable path therefore remains gated off while the destination-aware
pinned-UVA and CPU contingencies are evaluated.

The first pinned-UVA contingency allocator and GLM-shaped Marlin probe are now
complete. Final pinned backing stays on the local Grace NUMA node, produces
bit-exact Marlin output, and measured about 4.5% slower than HBM for the combined
gate/up plus down matrix sequence at batch one. See the
[Marlin/UVA experiment](experiments/2026-07-17-marlin-uva/README.md).

Phase 1 now has a header-only fail-closed manifest and deterministic EP4 expert
planner. It inventories all 175,527 tensors in about 1.5 seconds and separates
stored checkpoint bytes from the final fused-Marlin layout. Exact tracing of
non-routed tensors through TP4 sharding, dropped indexer copies, and runtime
fusion produces 4,181,609,280 resident bytes per rank. With the measured
machine capacities, the current deterministic plan places 3,097 experts in HBM
and 1,703 in pinned Grace memory per rank when retaining the baseline cache
allocation. Native 400K cache sizing now replaces that baseline input: the
host-main-cache scenario keeps 4,477 of 4,800 local layer-expert slots hot,
while the HBM-cache fallback keeps 3,425 hot. These final counts include exact
two-tier Marlin workspaces, maps, remap buffers, and one-expert conversion
scratch, plus conservative upper bounds for rounded baseline runtime metrics.
Both enforce the v2 plan's 5 GB HBM and 8 GB Grace reserves. Details are in the
[tier-plan experiment](experiments/2026-07-17-tier-plan/README.md).

The next loader prerequisite is also in place: safetensors iteration accepts a
fail-closed per-layer ownership map and skips remote packed weights, scales,
and metadata before payload materialization. Under linear EP4, this reduces the
planned checkpoint stream to 107,382,098,688 bytes per rank. It is tested as an
iterator primitive but is not yet wired into the tiered destination loader.

All dedicated v2 flags now flow through `EngineArgs` into a hashed
`TieredMoEConfig`. Cross-config validation enforces the pinned TP4/EP4, 400K,
batch-one, FP8 MLA, NUMA, reserve, and no-generic-offload contract. The real
`vllm serve --tiered-moe-plan-only` path now builds this engine config, validates
and hashes a checked-in GH200 machine profile, prints both complete physical
plans, and exits before sockets, workers, GPU allocation, or tensor payload
reads. Destination-loader wiring is the next implementation slice.

Phase 2 has started. The real `DefaultModelLoader` now installs the planner's
strict layer-aware EP4 ownership map and forwards it to safetensors. A compact
final-destination layout allocates component-major Marlin tensors from one HBM
buffer and one pinned-Grace buffer per layer, so weights, scales, and shape
metadata cannot split across tiers. A 60-hot/4-cold layer smoke test allocated
the exact 1,245,708,800 bytes in 0.683 seconds; all 256 sampled cold pages were
on the paired Grace NUMA node. A bounded production stager now rejects
interleaved or incomplete bundles, converts one real 19,464,240-byte checkpoint
expert with native Marlin, and commits the 19,464,200-byte result to Grace with
44,662,784 bytes peak HBM. Wiring this path into model parameter creation is
next. Worker startup now resolves the selected cache/expert scenario before
`initialize_model`, retains the exact rank plan in `DefaultModelLoader`, and
exposes it through a scoped construction context. The real rank-0 resolver
matches plan-only: 4,477 hot and 323 cold slots across 75 layers.

## Reproducing

The scripts expect this directory to be `agent_space/` inside the vLLM checkout
and the pinned model to be a sibling of that checkout under `models/`.

```bash
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:4 \
  --overlap --cpu-bind=none --unbuffered \
  bash run-cpu-offload-baseline.sh
```

After the health endpoint is ready, run `run-batch1-baseline.sh` or the commands
recorded in the result JSON files. Do not run model downloads from Booster.

## Layout

- `baseline/`: benchmark JSON, memory captures, reports, and diagnostic logs
- `jupiter-env.sh`: module, virtualenv, cache, and offline settings
- `run-cpu-offload-baseline.sh`: baseline server configuration
- `run-batch1-baseline.sh`: batch-one benchmark cases
- `benchmarks/`: focused hardware and kernel microbenchmarks
- `profiles/`: versioned physical machine profiles used by plan-only
- `experiments/`: dated raw results and experiment notes
- `gh200-vllm-w4a16-tiered-moe-plan-v2.md`: implementation plan beyond baseline
