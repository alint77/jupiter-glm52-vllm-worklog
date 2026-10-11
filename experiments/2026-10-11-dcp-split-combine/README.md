# Fused DCP tail: FlashMLA split combine + LSE reduce-scatter (not adopted)

2026-10-11. This is "seam 1" of the attention megakernel estimate. Under DCP4,
FlashMLA sparse decode writes fp32 split partials. Its combine kernel reduces
them to bf16 `out` + `lse` over all 64 gathered heads, and the one-shot
`lse_reduce_scatter` then pulls each rank's 16-head slice from the peers.

The idea was one kernel instead of two:

- **Phase A:** combine the splits into the custom-AR IPC buffer.
- **Release barrier.**
- **Phase B:** the existing warp-per-pair LSE reduce-scatter, reading the
  peers' buffers.

The estimate assumed the combine's ~6.5 µs kernel time sat on the critical
path, giving ~5-10 µs/layer (0.6-0.8 ms/step at 8 tokens).

## Result: slower at every size, and the ceiling is small

The fused kernel is bitwise-equal to the current path but slower. Measured
with `bench_split.py` (4 ranks, 78 graph-captured layers, max over ranks; the
two runs agree to ±0.2 µs):

| tokens | separate | fused | main only | main + combine | lse_rs alone |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.45 | 19.61 | 10.06 | 13.81 | 5.52 |
| 8 | 23.42 | 24.50 | 13.85 | 16.63 | 6.39 |
| 16 | 28.27 | 30.74 | 17.68 | 20.42 | 7.55 |
| 32 | 38.40 | 43.79 | 24.72 | 27.89 | 10.22 |
| 64 | 58.85 | 68.54 | 38.78 | 43.09 | 14.89 |

All values are µs/layer (`logs/b2.log`).

The estimate was wrong because FlashMLA launches its combine with PDL
(`csrc/smxx/decode/combine/combine.cu`: `cudaLaunchAttributeProgrammaticStreamSerialization`
plus `cudaGridDependencySynchronize`). The main kernel triggers launch
completion, so the combine overlaps the main kernel's tail. Its profiler
duration (5.9-7.2 µs) mostly counts waiting. The combine adds only **2.8 µs
at 8 tokens and 4.3 at 64**.

Fusion cannot remove the work of reading the fp32 partials. It can only
remove the combine→lse_rs launch gap and the bf16 `out` round trip (0.5 MB
at 8 tokens). That ceiling is ≤ ~1-2 µs/layer, about 0.1 ms/step, which is
below the served A/B noise.

The fused kernel loses because phase A runs on the 36 barrier blocks and
reads 6.3 MB of partials at 8 tokens (16.8 MB at 64). That is bandwidth-bound
on 36 SMs, where FlashMLA's combine uses the whole GPU. A grid-wide arrival
counter could give phase A every SM, but then phase A costs at least what the
combine costs now, so it would at best match the ceiling above. Not built.

Lesson: under PDL, a kernel's profiler time is not its critical-path cost.
Price a boundary by removing it from a graph (the "main only" vs
"main + combine" columns), not from kernel durations.

## What was built (kept on side branches)

- **FlashMLA** `dcp-split-partials` (`b4cd786`, local clone at
  `/e/fscratch/profound/$USER/src/FlashMLA`): adds `sparse_decode_fwd_partial`,
  which returns `o_accum`, `lse_accum` and `num_splits` and skips the combine.
  Build: cmake from the vLLM tree with `FLASH_MLA_SRC_DIR` and
  `TORCH_CUDA_ARCH_LIST=9.0a`, target `_flashmla_C`
  (`/e/fscratch/profound/$USER/build/fmla-partial`).
- **vLLM** `dcp-fused-tail` (`885cfed3c6`, worktree
  `/e/fscratch/profound/$USER/worktrees/dcp-fused`):
  - `split_lse_reduce_scatter_kernel`, which replicates FlashMLA's combine
    arithmetic exactly: per-lane max/sum shuffles, IEEE exp2f/log2f,
    sequential FFMA over splits, +inf LSE for empty rows;
  - `barrier_at_start_release`;
  - `VLLM_DCP_SPLIT_COMBINE` and the flashmla_sparse hook;
  - `test_one_shot_split_combine`: bitwise vs separate on 2 and 4 GPUs at
    1/8/64 tokens, with empty rows, eager and graph-replayed (passes).

The main tree and prod are unchanged.
