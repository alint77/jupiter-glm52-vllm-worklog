# DCP4 one-shot collectives at 64 tokens

2026-10-11. The c16 trace (2026-10-09-tp-sliced-experts, "c16 per-kernel SOL")
put the two per-layer DCP collectives at 35 µs (gather_cat, the MLA query
gather) and 19 µs (lse_rs, the attention combine) per call at 64 tokens, 78
layers each: ~4.2 ms of a 51 ms step. Both are one-shot pull kernels on vLLM's
custom all-reduce machinery: 36 blocks of 512 threads, `barrier_at_start`,
peers' data loaded over NVLink, `barrier_at_end`.

Result: commit `4524b36e4e` on `tp-sliced-moe`. Same 36 blocks and barriers,
bitwise-identical outputs:

| tokens | gather_cat old → new (µs) | lse_rs old → new (µs) |
| --- | --- | --- |
| 8 | 7.8 → 6.2 | 6.4 → 6.4 (old kernel kept) |
| 16 | 11.2 → 7.7 | 7.6 → 7.6 (old kernel kept) |
| 24 | 14.9 → 9.2 | 10.5 → 10.5 (old kernel kept) |
| 32 | 18.7 → 10.7 | 11.2 → 10.5 |
| 48 | 26.5 → 13.5 | 15.1 → 13.7 |
| 64 | 34.1 → 16.5 | 18.5 → 15.4 |

`bench_dcp.py`, graph-captured, 78 calls per graph, max over the 4 ranks
(`logs/b4-port.log`). The 64-token old numbers match the served trace, so the
microbench reproduces the served cost. Expected served effect: ~20 µs × 78 ≈
1.6 ms per step at c16 (~3%), ~0.65 ms at c8. The served A/B is below.

## What limited them

Measured in order. Each step's harness is in this directory.

1. **Loads in flight (`logs/b1.log`).** Unrolling gather_cat to 8-16
   independent 16 B loads per thread before the stores: 34 → 22 µs at 64
   tokens. 32-bit index math beats 64-bit (26.0 vs 28.0 at U=8). Warp-per-pair
   lse_rs (every rank's packs loaded before the LSE math, as the prefill
   combine does): 18.3 → 15.5, but slower below 32 tokens (fewer warps than
   pairs fill the grid).
2. **More SMs (`logs/b2.log`, timing-only probe).** Blocks past 36 skipping the
   barrier (unsafe, inputs static): gather_cat 15.8 µs at 132 blocks. lse_rs
   does not improve with blocks.
3. **Push, barrier-free (`push.cuh`, `bench_push.py`, `logs/p1.log`,
   `logs/p2.log`).** From NVIDIA's "Every µs Matters" (Shen et al.):
   - the sender stores into the peer's symmetric scratch, then raises a
     per-block release flag;
   - double-buffered halves, so a call's receipt is the next call's permission
     (no end barrier);
   - monotonic epoch flags (no start barrier, no resets).

   Bitwise-correct under rank drift: 20 rounds of 12 calls with per-rank
   spin delays. But at 64 tokens it gave 17.9 µs gather_cat and 16.3 µs lse_rs
   (23 µs before the peer interleave). At 8 tokens it saved only ~0.6 µs:
   barriers are not the dominant fixed cost here.
4. **Rank order (`logs/b3-il*.log`).** The old gather_cat loop is rank-major:
   a grid-stride pass covers a contiguous index range, so one source rank, so
   one NVLink carries the whole grid at a time. Alternating source ranks every
   32 units, unroll 8, stock 36 blocks: 16.7 µs at 64 tokens. That is as fast
   as push or 132 blocks, with no new flags or buffers. The marginal rate is
   now ~118 GB/s per peer (the GH200 quad's links are ~150 GB/s per direction
   per peer). The fixed cost is ~5.5 µs per call.

The paper's other levers don't fit:

- NVLS multicast: there is no NVSwitch on the GH200 quad.
- LL128 atomics: non-deterministic, which breaks "same math".
- Sentinel sync: q and out are arbitrary bf16, so no value can be reserved.
- Per-word LL flags: they halve bandwidth, and 64 tokens is bandwidth-bound.

## Shipped

- **`gather_cat_kernel`:** ranks interleaved every 32 uint4 units; 8 loads in
  flight per thread; 32-bit index math with a `units < 2^31` host check.
- **`lse_reduce_scatter_warp_kernel`:** used when `ngpus <= 4`,
  `D_packs == 64` and at least 512 pairs; otherwise the old kernel. The 8-GPU
  case is unmeasured, and its x[8][2] alone is 64 registers.
- **Tests** (`tests/distributed/test_custom_all_reduce.py`, 2 and 4 GPUs,
  pass):
  - gather_cat bitwise vs `torch.cat` + `dist.all_gather` at 8 and 64 MLA
    tokens, plus a ragged shape (60 units per rank, offset start);
  - lse_rs at 16 / 31 / 32 / 64 tokens bitwise vs the prefill combine, on both
    sides of the switch.
- **Astra review** (gpt-6-astra): no bugs. It confirmed the interleave mapping
  is a bijection and the arithmetic order is unchanged. Its suggestions were
  the ragged and threshold tests, integer-view equality and the 8-GPU guard,
  all adopted.

## Served A/B

`ab_launch.sh`: c16 config (MTP3, tp_sliced, 1.6M pool, reserve 4.0), frozen
worktrees `dcp-base` (`25c3ed40c4`) and `dcp-new` (`4524b36e4e`). Two nodes
(holds 2269099, 2269100), arms in opposite order on each; `conc_scale.py` at
n = 8 and 16, contexts 5K and 50K, 2 reps. torch.compile came from the seeded
cache (42-46 s). Same residency in all arms (13528 hot / 5672 cold per rank).

Decode step time, same-node pairs (new - base, mean of 2 reps per arm;
`ab_table.py`):

| context | requests | base ms | new ms | node 2269099 | node 2269100 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5K | 8 | 39.29 | 38.52 | -0.73 | -0.80 | **-0.77** |
| 50K | 8 | 41.06 | 40.51 | -0.49 | -0.61 | **-0.55** |
| 5K | 16 | 57.75 | 56.30 | -1.38 | -1.53 | **-1.46** |
| 50K | 16 | 59.74 | 58.98 | -0.38 | -1.14 | **-0.76** |

All 8 pairs favour the new kernel. At 8 requests the ~0.65 ms the microbench
predicts is met (-0.55 to -0.77). At 16 requests 5K is close to the 1.6 ms
prediction (-1.46); 50K is noisier (-0.38 / -1.14 between nodes). Acceptance
is unchanged (2.84-3.31 in both arms). Overall about **-1.1 ms/step (-2%) at
16 requests and -0.66 ms (-1.7%) at 8**.

Shipped to prod: the sliced Claude Code server (`claude-glm53-sliced-df2-dcp4.sh`)
was restarted on `4524b36e4e` the same day.
