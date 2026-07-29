"""Build a realistic agentic PyTorch/CUDA coding prompt suite.

The existing c1/q4 suite is 24 prompts of roughly 40 tokens each. Real agentic
coding turns are nothing like that: they carry file contents, profiler tables,
build errors and stack traces, so they run 1-4K input tokens and ask for a
concrete patch or diagnosis. Decode throughput is measured over the generated
answer, but the input length still sets the attention and KV work per step, so
the prompt shape has to be realistic for the number to mean anything.
"""

import json
import pathlib


REPO = pathlib.Path(__file__).resolve().parents[3]


def excerpt(rel, start, end, lang="cpp"):
    """Real source from this checkout, so the prompts carry authentic context.

    An agentic coding turn arrives with the file it is working on already in
    context; a 40-token question does not exercise the same attention or KV path
    at all.
    """
    lines = (REPO / rel).read_text().splitlines()[start - 1:end]
    body = "\n".join(lines)
    return f"`{rel}` lines {start}-{end}:\n\n```{lang}\n{body}\n```"


MARLIN_LAUNCH = '''```cpp
  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int sh_cache_size =
      get_kernel_cache_size(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel = get_marlin_kernel(...);

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr,
      zp_ptr, g_idx_ptr, sorted_token_ids_ptr, expert_ids_ptr,
      num_tokens_past_padded_ptr, topk_weights_ptr, top_k, mul_topk_weights,
      num_groups, prob_m, prob_n, prob_k, locks, has_bias, use_atomic_add,
      use_fp32_reduce);
```'''

SPLIT_K = '''```cpp
  int parallel = num_tokens_past_padded / moe_block_size;
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;

  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }
  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);
```

and the cross-CTA reduction barrier:

```cpp
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}
```'''

TRACE_TABLE = '''```
| Category                     |   cum |  union |  solo | share | n/step |
| Routed W4 Marlin (MoE)       | 42.31 |  25.27 | 20.15 | 32.3% |    300 |
| TP custom all-reduce         |  9.27 |   9.27 |  9.27 | 14.8% |    166 |
| GPU idle                     |     - |      - |  7.95 | 12.7% |      - |
| Dense/shared cutlass + nvjet |  5.02 |   5.02 |  4.94 |  7.9% |    502 |
| Glue (elementwise/triton)    |  4.92 |   4.82 |  4.65 |  7.4% |  2,410 |
| DCP NCCL AllGather/RS        |  3.12 |   3.12 |  3.12 |  5.0% |    271 |
| Sparse MLA (FlashMLA)        |  2.65 |   2.37 |  2.37 |  3.8% |    162 |
```'''

OOM_TRACE = '''```
[rank2]: torch.AcceleratorError: CUDA error: an illegal memory access was encountered
[rank2]:   File "vllm/model_executor/layers/fused_moe/modular_kernel.py", line 1663, in apply_tiered
[rank2]:     cold_output = run_tier(1)
[rank2]:   File "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py", line 191, in _fused_marlin_moe
[rank2]:     output = ops.moe_wna16_marlin_gemm(
[rank2]: CUDA kernel errors might be asynchronously reported at some other API call
```'''

NCU_TABLE = '''```
  dynSmem  occ/SM soloA_ms soloB_ms   serial union_ms ov_frac   peak
    76800       3 1768.806  900.821 2669.627 2381.854    0.39      3
    57344       4 1769.038  899.633 2668.671 2292.612    1.00      4
    48000       4 1770.202  901.475 2671.677 2289.643    1.00      4
```'''

GRAPH_ERR = '''```
RuntimeError: CUDA error: operation would make the legacy stream depend on a
capturing blocking stream
  File "vllm/v1/worker/gpu_model_runner.py", line 2841, in capture_model
    self._dummy_run(num_tokens, capture_attn_cudagraph=True)
  File "torch/cuda/graphs.py", line 185, in capture_end
    super().capture_end()
```'''

PROMPTS = [
    ("cuda-occupancy", f"""I am looking at vLLM's Marlin MoE launch path and something about occupancy
does not add up. Here is the launch site:

{MARLIN_LAUNCH}

On a GH200 the device reports 233472 bytes of shared memory per SM and 232448
bytes per block opt-in, and the profiler shows this kernel launching with grid
396, 128 threads, and 76458 bytes of dynamic shared memory per CTA, while
`get_kernel_cache_size` for the selected tile returns 25856.

Walk me through what actually determines the achieved CTAs per SM here, what the
consequence is if a second kernel is launched concurrently on another stream,
and whether changing `blocks_per_sm` can fix it. If it cannot, propose the
minimal change to the launch that would, and say what could break."""),

    ("cuda-splitk", f"""Explain what this Marlin work-partitioning code does when the grid is much
larger than the available MN tiles, and what it costs:

{SPLIT_K}

Concretely: an expert-parallel MoE tier has 1 activated expert, prob_n is 4096,
thread_n is 128, prob_k is 6144, thread_k is 64, and the launch uses a grid of
132 CTAs. How many MN tiles are there, how many ways does K get split, how many
CTAs end up cooperating per output tile, and what synchronisation and extra
memory traffic does that imply? Then tell me what grid you would pick instead
and why, and how the answer changes if 5 experts are activated instead of 1."""),

    ("profile-readout", f"""Here is a per-step critical-path budget from a four-rank decode profile at
396K context, concurrency 4, MTP3:

{TRACE_TABLE}

The step wall is 62.43 ms. I want to prioritise optimisation work. For each of
the top four rows, tell me whether it is likely to be reducible and by what
mechanism, and flag any row where the number is probably measuring something
other than what its name suggests. Be specific about what additional
measurement would settle each case."""),

    ("debug-illegal-access", f"""A tiered MoE run crashes on one rank only:

{OOM_TRACE}

Context: the model runs two Marlin kernels per layer on two CUDA streams, one
reading weights from HBM and one from pinned host memory through UVA. Both
kernels were given a shared `workspace` tensor to save memory. The crash is not
deterministic and sometimes presents as a hang at 100% GPU with the host stuck
in `cudaStreamSynchronize`.

Diagnose the most likely root cause, explain the mechanism precisely, and give
me the fix. Also explain why this would present as a hang in some runs and an
illegal access in others."""),

    ("interpret-probe", f"""I wrote a probe that launches two identical kernels on two streams and varies
only the dynamic shared memory requested per CTA. `peak` is the maximum number
of CTAs resident on any single SM across both kernels, from `%smid` and
`%globaltimer` recorded per CTA:

{NCU_TABLE}

Interpret this. What is the hardware rule being hit, why does `ov_frac` jump
from 0.39 to 1.00 between the first and second rows, and why does the union time
barely improve even when the CTAs do become co-resident? What would you measure
next to find out whether the two kernels are actually sharing bandwidth
productively?"""),

    ("graph-capture", f"""CUDA graph capture fails when I add a second stream to my MoE layer:

{GRAPH_ERR}

The layer does: fork an aux stream with `aux.wait_stream(main)`, run the cold
expert tier on aux, run the hot tier on main, then `main.wait_stream(aux)` and
add the two outputs. It works fine in eager mode.

Explain the capture rules I am violating, show the corrected sequence, and tell
me what warm-up is required before capture. Then explain why timing this
fork/join in eager mode gives a very different answer from replaying it inside a
graph, and which one I should trust for a production estimate."""),

    ("pytorch-perf", """Profile this PyTorch inference path and tell me what to fix:

```python
class ExpertMLP(nn.Module):
    def __init__(self, hidden, inter, n_experts):
        super().__init__()
        self.w13 = nn.Parameter(torch.empty(n_experts, 2 * inter, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, hidden, inter))

    def forward(self, x, topk_ids, topk_w):
        out = torch.zeros_like(x)
        for e in range(self.w13.shape[0]):
            mask = (topk_ids == e).any(dim=-1)
            if not mask.any():
                continue
            xe = x[mask]
            gate, up = (xe @ self.w13[e].t()).chunk(2, dim=-1)
            h = F.silu(gate) * up
            out[mask] += (h @ self.w2[e].t()) * topk_w[mask].sum(-1, keepdim=True)
        return out
```

At batch 4 with 256 experts and top-8 routing this is far slower than the fused
kernel it replaced. Identify every distinct source of overhead, rank them, and
rewrite the forward to remove the top two. Keep the numerics equivalent and say
where they are not exactly equivalent."""),

    ("numerics", """I changed the launch grid of a split-K GEMM and the output moved by up to
0.00195 in bfloat16, though every individual run is exactly reproducible.
Downstream I have a golden SHA-256 over a 256-token greedy continuation that now
mismatches.

Explain why the grid change moves the result at all, why the change is
deterministic per configuration but different between configurations, and
whether this constitutes a correctness regression. Then tell me how you would
re-establish a defensible correctness gate that is robust to this class of
change without becoming so loose that it stops catching real bugs."""),

    ("kernel-write", """Write a CUDA kernel for GH200 (sm_90a) that measures whether two concurrently
launched kernels are genuinely co-resident on the same SM, rather than merely
overlapping in wall-clock time.

Requirements: record per-CTA start and stop from `%globaltimer` and the SM id
from `%smid`; take the dynamic shared memory request as a launch parameter so
occupancy can be swept; run on two non-blocking streams with no cross-stream
event waits; and post-process to report the peak number of CTAs resident on any
single SM across both kernels.

Give the full kernel plus the host driver, and explain why time-range overlap of
the two kernels is not evidence of co-residency."""),

    ("c2c-bandwidth", """On a GH200 superchip I am streaming quantised expert weights from pinned host
(Grace LPDDR) memory into a GEMM over NVLink-C2C, while a second GEMM streams
different weights from HBM.

Measured: the HBM kernel alone sustains about 3.1 TB/s, the C2C kernel alone
about 410 GB/s. Run concurrently with genuine co-residency, neither slows down
measurably. Run concurrently without co-residency, the union is essentially the
sum.

Explain what this tells me about where the bottleneck is and is not. Then design
an experiment that would distinguish L2 capacity thrashing from memory-fabric
saturation from pure scheduling serialisation, naming the specific hardware
counters and the expected signature of each."""),

    ("moe-placement", """I have an expert-parallel MoE where each rank owns 64 of 256 routed experts,
and per rank about 3,000 experts are resident in HBM with the remainder read
from host memory over C2C. A profile shows the per-layer span is
`sum over layers of max over ranks`, and that 86% of the cross-rank skew is an
order statistic rather than persistent imbalance.

I want to reduce the skew. Evaluate these options and tell me which are dead on
arrival and why: rebuilding the static owner map against expected activation
cost; moving more activation mass to the host tier; replicating the hottest
experts on all ranks and assigning them per step to the least-loaded rank;
reducing the number of synchronisation points. For any option you do not reject,
sketch the implementation and the cheapest offline test that would falsify it."""),

    ("build-debug", """A CUDA extension fails to link:

```
undefined reference to `void marlin_tight::Marlin<1125899906909960l, ...,
  128, 1, 4, 8, false, 4, 8, false>(int4 const*, int4 const*, int4*, ...)'
```

The template is declared in a header, explicitly instantiated in one .cu file,
and its address is taken in a different .cu file through a big if/else selector.
It links fine in the upstream build but not in my standalone build.

Explain the mechanism, name the nvcc flag that upstream is almost certainly
passing, and explain what that flag changes about the emitted symbols. Then tell
me how I would have found this without guessing, using only `nm`."""),

    ("agentic-plan", """I am optimising a 361 GiB W4A16 MoE model served on one GH200 node at up to
400K context. Current state: decode is about 108 tok/s at concurrency 1 with
3-deep speculative decoding, and about 180 tok/s aggregate at concurrency 4.

A profile says: routed MoE Marlin is 32% of the step, the tensor-parallel
all-reduce 15% (almost entirely expert-load skew), small-kernel scheduling
overhead 12%, and all context-length-specific work only 12%.

Give me a prioritised plan. For each item state the expected gain with its
reasoning, the cheapest experiment that could refute it before any cluster time
is spent, and the risk. Be explicit about which items are bets versus which are
mechanical. Do not propose anything whose payoff you cannot estimate."""),

    ("review-patch", """Review this patch for correctness and for anything that would only show up in
production:

```python
_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}

def _apply_tier_launch_policy(kernel, tier_name, device):
    from ...marlin_moe import MarlinExpertsBase, MarlinLaunchPolicy
    impl = getattr(kernel, "impl", None)
    if not isinstance(impl, MarlinExpertsBase):
        return
    blocks_per_sm = _TIER_BLOCKS_PER_SM.get(tier_name)
    if blocks_per_sm is None:
        return
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    impl.launch_policy = MarlinLaunchPolicy(
        smem_mode=ops.MARLIN_SMEM_TIGHT, grid_blocks=sms * blocks_per_sm)
```

Context: `kernel.impl` is a modular-kernel wrapper whose `fused_experts`
attribute holds the actual Marlin experts object that the GEMM reads its
configuration from. The policy is meant to apply only to small decode batches;
large prefill batches should keep the default launch. Point out every defect and
give the corrected version."""),

    ("dcp-attention", """I am porting decode-context parallelism to a sparse MLA attention backend so
that a 400K context can be sharded across 4 ranks instead of replicated.

Each rank ends up with a subset of the KV blocks and computes a partial
attention output plus a log-sum-exp, which then has to be combined across ranks.
One rank can legitimately end up with no valid blocks for a given query.

Explain the correct combine formula, what sentinel the empty-shard case must
produce for the combine to stay finite, and why returning +inf for that case
silently poisons the result. Then describe the unit test you would write to
catch it, given that the failure only appears at long context with a specific
block distribution."""),

    ("speculative", """With 3-deep MTP speculative decoding my per-position acceptance is 84%, 65% and
46%, giving about 2.96 accepted tokens per target step.

I am comparing two builds and one shows both higher throughput and higher
acceptance. Explain why comparing raw output tokens per second between them is
misleading, derive the quantity I should compare instead, and show the algebra
that separates a kernel speed-up from an acceptance change.

Then: if the only difference between the builds is a floating-point reduction
order that moves logits by about one bfloat16 ulp, is a systematic 2-point
acceptance shift plausible or does it indicate a real bug? Justify your
answer."""),
]


CONTEXT = {
    "cuda-occupancy": [
        ("csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu", 184, 260, "cpp"),
        ("csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu", 262, 350, "cpp"),
    ],
    "cuda-splitk": [
        ("csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", 370, 470, "cpp"),
    ],
    "debug-illegal-access": [
        ("vllm/model_executor/layers/fused_moe/modular_kernel.py", 1582, 1700, "python"),
    ],
    "review-patch": [
        ("vllm/model_executor/layers/fused_moe/experts/marlin_moe.py", 57, 130, "python"),
    ],
    "graph-capture": [
        ("vllm/model_executor/layers/fused_moe/modular_kernel.py", 1600, 1670, "python"),
    ],
    "pytorch-perf": [
        ("vllm/model_executor/layers/fused_moe/experts/marlin_moe.py", 130, 245, "python"),
    ],
    "moe-placement": [
        ("vllm/model_executor/model_loader/tiered_moe_execution.py", 1, 100, "python"),
    ],
    "dcp-attention": [
        ("vllm/v1/attention/backends/utils.py", 1, 80, "python"),
    ],
}


def main():
    out = pathlib.Path(__file__).resolve().parent / "agentic-prompts.jsonl"
    with out.open("w") as f:
        built = []
        for i, (name, text) in enumerate(PROMPTS):
            ctx = ""
            for rel, a, b, lang in CONTEXT.get(name, []):
                try:
                    ctx += "\n\n" + excerpt(rel, a, b, lang)
                except (OSError, IndexError):
                    pass
            full = (text + ctx) if not ctx else (
                "Here is the relevant source from the checkout I am working in."
                + ctx + "\n\n" + text)
            built.append(full)
            f.write(json.dumps({
                "id": f"agentic-{i:02d}-{name}",
                "domain": "agentic-cuda-pytorch",
                "output_tokens": 512,
                "prompt": full,
            }) + "\n")
    lens = [len(t) for t in built]
    print(f"wrote {len(PROMPTS)} prompts to {out}")
    print(f"chars: min {min(lens)}, median {sorted(lens)[len(lens) // 2]}, "
          f"max {max(lens)}")


if __name__ == "__main__":
    main()
