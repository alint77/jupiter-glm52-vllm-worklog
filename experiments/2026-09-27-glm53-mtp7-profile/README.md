# GLM-5.3 W4A16, MTP K=7, c=1, 400K: fresh decode profile and the INT4 one-kernel port

Back on GLM after the MiMo work (`2026-09-27-dak-tiered-moe-kernel`). Every
GLM breakdown so far is MTP3 at 96K and predates the MiMo-era changes, so the
first step is a new profile of the shape we now want to serve:

- **MTP K=7, not DFlash2.** DFlash2 is overfit to GSM8K-like text (5.6 AL there
  against 3.10 on Claude-Code-shaped traffic, `2026-09-04-spec-comparison`),
  runs eager because its captured graph loses acceptance, and needs an
  unbudgeted draft KV cache. MTP7 measured 3.24 AL on the same traffic.
  It verifies 8 tokens per step, the same shape as MiMo's DFlash K=7.
- c=1, DCP1, 400K context, reserve 10, cold prefetch at 1024 tokens, prefix
  caching off so a repeated prompt really prefills.

## Files

- `serve.sh` -- the server (the spec-comparison mtp7 arm plus cold prefetch)
- `capture.py` -- decode profiles at short, 96K and ~290K context, each with an
  unprofiled control on the same request
- `run.sh` -- serve + capture on a held node (`hold.sbatch`, `onnode.sh`)
- analysis: `../2026-09-26-mimo-decode-profile/{analyze,imbalance,gaps}.py`,
  which now also separate out the DSA indexer

## Sanity check of the analyzer on the old MTP3 trace (job 1665068, 96K)

| target graph (ms/step) | |
|---|---|
| cold Marlin (Grace) | 9.94 |
| hot Marlin (HBM) | 1.97 |
| TP all-reduce | 7.09, of which **6.50 waiting** for the slowest GPU |
| dense GEMM | 3.71 |
| norm/rope/elementwise | 3.65 |
| attention (sparse MLA) | 1.84 |
| MoE routing/align/sum/act | 1.48 |
| DSA indexer | 0.46 |

## INT4 one-kernel decode MoE

The one-kernel decode path (`tiered_decode.cu`) was MXFP4-only. GLM-5.3 W4A16
has the same expert shape (6144 x 2048, top-8), so the port is the weight
format only:

- weights: the same Marlin 4-bit repack, so the same nibble order; uint4b8
  decodes exactly as `hfma2(0x6400 | code, 2^-14, -1032 * 2^-14)`, landing at
  the `value * 2^-14` the MXFP4 decode uses, so nothing downstream changes
- scales: bf16 in `marlin_permute_scales` order, where a lane's rows g and
  g+8 of block `mb` are one 32-bit word at element `8g + 2mb`; a stage carries
  4 KB of scales instead of 2 KB (215 KB of shared memory)
- the path now also runs with replicas off (static slot maps) and with a
  shared expert the runner executes itself (GLM's case)
- tests: `tests/kernels/moe/test_tiered_decode_moe.py` runs both formats;
  INT4 max error 2.9e-3 to 3.1e-3 of the row max against Marlin's 6.0e-3 to
  7.6e-3 (T = 1, 3, 8), MXFP4 unchanged. vLLM commit `12a52eacc7`.

## Results (jobs 2096362, 2096391; MTP7, c=1, 400K, reserve 10)

Unprofiled step time, 6 s windows on sampled (T=1) text. **These windows move
by +-3-5 ms with the generated content**, so only same-node rows compare, and
only coarsely; the greedy A/B below is the controlled measurement.

| node | arm | short | 96K | 288K |
|---|---|---|---|---|
| 2096391 | Marlin | 45.1 | 45.8 | 46.2 |
| 2096391 | INT4 one-kernel | 45.8 | 45.8 | 45.5 |
| 2096362 | Marlin | 46.2 | 40.6 | 41.1 |

Context length does not move the step: 288K costs what short does.

### Where a step goes (profiled, Marlin, short; profiler adds ~25%)

| bucket | ms/step |
|---|---|
| cold Marlin (Grace) | 17.5 (hot+cold union 20.5) |
| hot Marlin (HBM) | 2.6 |
| waiting for the slowest GPU after MoE | 8.0 -- **cold** spread 8.07, hot 1.63 |
| MTP7 drafting | 8.9, of which ~3.3 GPU work |
| dense GEMM | 3.9 |
| attention (sparse MLA) | 2.0 |
| MoE routing/align/sum/act | 1.5 |
| DSA indexer | 0.28 short, 0.58 at 96K, 1.28 at 288K |

1. **The decode MoE is bound by reading cold experts over C2C.** 17.5 ms of
   cold Marlin over 75 layers is ~233 us per layer, ~4 cold experts (20.3 MiB)
   per rank per layer at ~380 GB/s. MiMo reads 1-3.
2. **The waiting is cold imbalance**, rotating across all four ranks (22-28%
   last to arrive each). Replicas with balancing are the lever; the GLM
   replica campaign (`2026-09-05-decode-placement-replicas`) measured -9%.
3. **The INT4 one-kernel path does not change end to end.** With PDL off its
   MoE is 19.7 ms against Marlin's ~22.0 (w13 12.5, w2 6.8, small kernels
   0.5), but the time saved turns into waiting for the slowest rank. It is a
   prerequisite for in-kernel time balancing, not a win by itself.
4. **The DSA indexer does grow with context** (0.28 -> 1.28 ms), but is small.
5. Prefill ran at **9,565 tok/s** (96K in ~10 s), against 4,145 tok/s in the
   09-04 MTP3 profile.

### MTP drafted eagerly: a capture-size gap, not a choice

Only 2 CUDA graph launches per step, ~430 eager kernel launches, and a
drafting phase that is ~65% GPU idle. The autoregressive speculator captures
two routines: the draft prefill (position 0, K+1 tokens) and a separate graph
for the draft decodes (positions 1..K-1, **one token each**). The decode
manager only captures sizes from `cudagraph_capture_sizes` that fit
`max_num_reqs x 1` (`v1/worker/gpu/cudagraph_utils.py:229-236`). At c=1
with `cudagraph_capture_sizes: [8]` there are none, so the graph is skipped
without a word and six of the seven MTP passes run eager every step.

The c=1 MTP profiles before this one had the same gap (the 09-04 MTP3 profile
captured `[4]`); the c=4 production launcher escapes it because
`[4, 8, 12, 16]` contains 4 = max_num_seqs x 1.

Fix: `cudagraph_capture_sizes: [1, 8]` (`serve.sh`, `CAPTURE_SIZES`). Job
2096391, short context, profiled: graph launches 2 -> 8 per step, eager
launches ~430 -> ~63, drafting 8.3-10 -> 3.3 ms, GPU idle ~5 -> 1.6 ms.

### Reserve 7 instead of 10

The 10 GB reserve was sized for DFlash2's unbudgeted draft KV; MTP has none.
Reserve 7 boots (observed free 7.5 GiB against a 5.6 minimum) and keeps
**2427 hot experts per rank instead of 2284** (+143).

## Greedy A/B: `[8]` against `[1, 8]` (jobs 2096692, 2096693)

`ab.sh` + `bench.py`: two nodes, arms in opposite orders, server restarted per
arm; per arm three greedy runs of ~1450 tokens on the short and the 96K
prompt, step time from the engine counters over the decode.

**Greedy decoding is not reproducible here** (already noted for GLM-5.3 in
`2026-08-29-glm53-w4a16`): within one arm tokens per step ranges 4.0-7.0. And
step time *tracks* it, so arms only compare at matched acceptance. Least
squares over all 24 runs, `step_ms ~ tokens_per_step + arm + node + context`:

| term | ms/step |
|---|---|
| **`[1, 8]` vs `[8]`** | **-0.65 +- 0.28** |
| per accepted token per step | **+2.02 +- 0.19** |
| 96K vs short | -1.80 +- 0.30 |
| node 2096693 vs 2096692 | +0.45 +- 0.28 |

- Capturing the draft-decode graph is real but small, ~1.4%. The profile's
  ~5 ms of drafting idle was mostly profiler cost on ~430 eager launches;
  unprofiled, the host launches ahead of the GPU. Profiled idle time for
  eager code is not a usable estimate here.
- **Open: step time rises ~2 ms per extra accepted token.** The verify is
  always 8 tokens and the draft always 7 passes, so something per accepted
  token -- host-side output handling, or routing that differs with the
  content that accepts well -- costs ~2 ms. Not yet explained; it is larger
  than any other effect in the table.
- The earlier one-kernel vs Marlin comparison used single sampled windows and
  is inside this spread; it needs the same treatment before being quoted.
