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
and 1,703 in pinned Grace memory per rank. Details are in the
[tier-plan experiment](experiments/2026-07-17-tier-plan/README.md); exact
cache-tier and workspace sizing are the next planner slice.

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
- `experiments/`: dated raw results and experiment notes
- `gh200-vllm-w4a16-tiered-moe-plan-v2.md`: implementation plan beyond baseline
