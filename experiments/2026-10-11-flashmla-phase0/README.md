# FlashMLA sparse decode: where the time goes (Phase 0)

2026-10-11. FlashMLA's sparse fp8 decode (V3.2 KV, h64) at the DCP4 decode
shape: q [1, T, 64, 576], 768-wide index rows with 470-540 live keys
compacted to the front. Before this, the cost model was "~13 µs fixed +
~4.3 µs per 64-key block", which turned out to be wrong in its block counts.

The probes are compile-time macros on FlashMLA branch `phase0-probes`, all
off by default:

- `FMLA_TRACE`: per-CTA globaltimer/clock64 stamps; the buffer is passed via
  env `FMLA_TRACE_PTR`.
- `FMLA_CONST_FILL`: the producer stores constant K instead of gathering and
  dequantizing.
- `FMLA_PDL`: programmatic dependent launch of the main kernel.
- `FMLA_REG0/1/2`: the `setmaxnreg` split.

`build_variant.sh` recompiles `api.cpp` and the h64 instantiation with the
defines and relinks.

## 1. The planner leaves splits unbalanced (`bench_sched.py`, `logs/s1-*.log`)

For b = 1 (all T tokens share one schedule), `get_mla_metadata_kernel` gives
each part `ceil((nb + 5) / parts) + 5` blocks, where 5 is
`fixed_overhead_num_blocks`, and `parts = 132 / T`. At width 768 (12 blocks)
that yields:

| T | planner splits | longest | balanced alternative |
| ---: | --- | ---: | --- |
| 8 | 2+2+2+2+2+2 | 2 | (already fine) |
| 16 | 3+3+3+3 | 3 | 2 x 6 / 7 splits |
| 32 | 5+5+2 | 5 | 3+3+3+3 |
| 48, 64 | 9+3 | 9 | 6+6 |

Hand-built `DecodingSchedMeta` (graph replay, 78 layers, µs/layer
main+combine). Outputs differ only by fp32 summation order (max |diff|
4.9e-4, i.e. 1 bf16 ulp):

| T | planner, W 768 (prod) | best balanced, W 768 | best, W 576 (early stop at max live) |
| ---: | ---: | ---: | ---: |
| 8 | 15.67 | 15.61 (6 splits) | 14.61 (9 splits) |
| 16 | 19.74 | 18.24 (7) | 17.03 (5) |
| 24 | 23.48 | 21.73 (5) | 20.40 (4) |
| 32 | 27.58 | 24.25 (4) | 21.81 (4) |
| 48 | 39.84 | 32.33 (2) | 27.80 (2) |
| 64 | 42.95 | 34.50 (2) | 30.50 (2) |

Per step (x78): balancing alone saves **0.66 ms at 64 tokens (c16)** and
**0.26 ms at 32 (c8)**. Adding early stop at the step's max live count
(W = 576 here) saves another 0.31 / 0.19 ms. At 8 tokens there is ~0.08 ms,
from early stop only.

For b = 1 the schedule is static per (T, width), so balancing needs no
FlashMLA change: vLLM can pass its own metadata. Early stop needs the
metadata built on device from max(valid_counts) per indexer layer.

## 2. Cost model from the trace (`logs/p-trace.log`)

One CTA, one 64-key block (T = 1, 12 splits), stock kernel:

| phase | µs from CTA entry |
| --- | ---: |
| barrier init, schedule read | 0.3-0.8 |
| Q tile in smem | 1.0 |
| producer: gather + dequant of block 0 | ~1.9 (done at 2.6-2.75) |
| WG0: QK + softmax + PV of block 0 | ~2.8-3.0 (done at 5.4-5.7) |
| epilogue (fp32 o_accum or bf16 out, LSE, TMA store) | 2.1-3.4 (more with 128 CTAs: 3.6-4.6) |
| CTA end | ~9.1 |

Graph per-layer time is ~1.2 µs above the kernel span (launch and gaps).
Each further block adds ~3.0-3.1 µs. WG0 is the bottleneck: the producer
finishes block j+1 before WG0 is done with block j. CTAs all start within
0.2 µs, so launch spread is not a factor.

## 3. Constant fill: loading keys is not the bottleneck (`logs/p-const.log`)

Same schedules, K replaced by constants. All values are main-only µs/layer:

| config | stock | const fill | per-block slope stock -> const |
| --- | ---: | ---: | --- |
| T=1, 12 blocks serial in 1 CTA | 42.33 | 37.66 | 2.98 -> 2.65 µs |
| T=64 bal-2 (6 blocks, 128 CTAs) | 30.22 | 24.63 | ~4.1 -> ~3.2 µs |
| T=8 planner (2 blocks) | 13.04 | 11.97 | |

Loading is ~11% of a block on a lone CTA and ~22% at 128 CTAs (gather
contention). By Astra's model (2026-10-11-flashmla-token-packing),
a const-fill slope above ~3.3 µs/block, or a loading share around 0.25,
means token packing cannot pay. **Token packing is ruled out.**

## 4. PDL on the main kernel (`logs/p-pdl.log`)

Bitwise-identical outputs (checksums match stock). Main-only saves 0.5-1.5
µs/layer; main+combine saves 0-0.6 µs/layer (≤ 0.05 ms/step). Small, but
free once FlashMLA is patched anyway.

## 5. Serialized wgmma (the consumer's real cost)

ptxas warns C7512 ("wgmma.mma_async instructions are serialized due to
insufficient register resources") for this kernel in the stock build. The
SASS has all 44 HGMMAs (QK 36, PV 4+4) each fenced by its own
`WARPGROUP.ARRIVE` / `WARPGROUP.DEPBAR`. The `const_trace` build happened to
compile without serialization (3 DEPBARs). Its 12-block chain ran 27.2 µs vs
37.7 µs for `const`: ~2.0 µs per block instead of ~2.65-3.0.

`sweep_regs.sh` recompiles with other `setmaxnreg` splits (WG0 consumer /
WG1 PV / producer; the sum must be ≤ 504 = 3 x the launch-time 168,
because setmaxnreg only moves registers within the CTA). Only WG0 = 208 with
WG1 >= 160 removes the serialization: **208/160/136** and **208/168/128**,
with no local memory. Stock is 192/160/152.

(Pitfall, recorded: the first sweep allowed sums of 512. Those variants hang
in WG0's `setmaxnreg.inc`, and a grep that also matched the link line
overwrote the stock library with one of them. Restored from the
`dcp-fused` worktree copy, md5 b5843aa9.)

`bench_probe.py`, four GPUs of one node in parallel (`logs/r-*.log`).
Main+combine µs/layer, main-only in brackets. Every variant's output is
bitwise-identical to stock (same checksums):

| T | schedule | stock | r160 (208/160/136) | r168 (208/168/128) | r160 + PDL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | bal-1 (12 blocks, 1 CTA) | 42.27 (42.34) | 36.71 (36.67) | 36.60 (36.84) | 35.75 (35.80) |
| 4 | planner (12 x 1) | 13.59 (9.94) | 12.78 (8.89) | 12.87 (8.85) | 12.40 (8.10) |
| 8 | planner (6 x 2) | 15.59 (12.92) | 14.88 (11.87) | 14.61 (11.78) | 14.41 (11.13) |
| 8 | planner, W 576 (9 x 1) | 14.53 (11.15) | 13.33 (9.75) | 13.43 (10.00) | 13.35 (9.96) |
| 16 | planner (4 x 3) | 19.84 (17.36) | 18.47 (16.18) | 18.46 (15.88) | 18.51 (15.21) |
| 16 | bal-7 | 18.27 (14.62) | 17.08 (13.63) | 16.86 (13.32) | 17.36 (12.70) |
| 32 | planner (5+5+2) | 27.11 (24.59) | 25.90 (22.95) | 25.58 (22.37) | 25.79 (21.85) |
| 32 | bal-4 | 24.66 (19.27) | 22.88 (17.69) | 22.36 (17.58) | 23.04 (17.57) |
| 32 | bal-4, W 576 | 22.38 (18.36) | 20.63 (16.65) | 20.41 (16.63) | 21.57 (15.93) |
| 64 | planner (9+3) | 42.94 (38.38) | 41.95 (37.15) | 41.88 (37.25) | 42.10 (36.44) |
| 64 | bal-2 | 34.93 (30.18) | 32.79 (28.04) | 32.88 (27.92) | 32.98 (27.34) |
| 64 | bal-2, W 576 | 31.03 (26.05) | 29.35 (24.86) | 29.36 (24.65) | 29.66 (24.23) |

The serial chain drops from 2.98 to 2.58 µs per block, not to the ~2.0 of
`const_trace`. With the consumer no longer serialized, part of the
producer's gather (~0.6 µs/block) is now exposed. In serving configurations
the fix is worth 0.7-2.1 µs/layer (0.06-0.17 ms/step). PDL on top helps the
main kernel but not consistently main+combine (the combine already overlaps
it), so it is not worth carrying.

## Summary: per-step saving vs prod (planner, W 768, stock), x78 layers

| tokens (MTP3) | requests | prod µs/layer | balanced splits | + early stop (W 576) | + registers (r160) | total ms/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 15.59 | 0 | -1.06 | -1.20 | **-0.18** |
| 16 | 4 | 19.84 | -1.57 | ~-1.2 | ~-1.2 | **~-0.31** |
| 32 | 8 | 27.11 | -2.45 | -2.28 | -1.75 | **-0.51** |
| 64 | 16 | 42.94 | -8.01 | -3.90 | -1.68 | **-1.06** |

The T = 16 early-stop and register columns are estimates from `logs/s1-g1.log`
(stock W 576 bal-5: 17.03) and the r160 deltas. All of it is the same math:
balanced splits and early stop change only the fp32 split boundaries
(1 bf16 ulp, as with width 768); the register split is bitwise.

What remains per block (~2.6 µs after the register fix) is WG0's QK ->
softmax -> PV chain plus ~0.6 µs of exposed gather. The ~6 µs fixed part is
~1 µs of setup to Q-in-smem, ~2-4.5 µs of epilogue (o_accum fp32 128 KB per
CTA, LSE, TMA store wait), and ~1.2 µs of launch/gap per kernel.

## Phase 1 proposal (not started)

1. **Balanced schedule from vLLM.** For b = 1 the schedule depends only on
   (T, number of blocks), so vLLM can build the metadata itself and skip
   FlashMLA's planner. No FlashMLA change is needed.
2. **Early stop:** set the number of blocks from max(valid_counts) per
   indexer layer, on device (a tiny kernel, or graph-safe torch ops, writing
   the metadata). Keys past each row's live count are already -1 and masked,
   so skipping blocks that are -1 for every token is exact.
3. **Register split 208/160/136** in FlashMLA (`setmaxnreg`): a 3-line
   change in our FlashMLA build, bitwise-identical.

Then a served same-node A/B at c=2/8/16 and the usual acceptance check.
Token packing and PDL are dropped on these numbers.
