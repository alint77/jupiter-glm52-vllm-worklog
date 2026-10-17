# Kernel verification on the login node (2026-10-17)

Booster is unavailable (cluster full), so this phase uses only the login-node
GH200 GPU and CPU. The question: what can check our Hopper kernels' sync
(TMA + mbarrier rings, named barriers, split-K atomics, cross-CTA flags)
beyond the correctness tests? Kernels: `skinny_gemm.cu`, `sliced_decode.cu`,
`tiered_decode.cu`, `tiered_prefill.cu`.

## Result

- **compute-sanitizer** racecheck, synccheck and memcheck are clean on all four
  kernels' correctness suites. Planted bugs show those tools are **blind to TMA
  ring hazards**: a consumer that releases its stage before reading it (TMA
  overwrites live data, wrong output) passes both racecheck and synccheck.
- **SentryWard** (60a1ebc, built for Blackwell) runs on our sm_90a PTX after
  three PTX-level workarounds. It **proves skinny_gemm's in-CTA TMA/mbarrier
  pipeline race-free and well-synchronized** at two concrete launches, and
  **catches both ring mutants**, including the one the sanitizer misses.
- Limits: SentryWard does not model `wgmma` (tiered_prefill) or global
  `atom.add` (skinny split-K, the decode kernels' work stealing and
  last-done counters), and needs every control branch static. Nothing else
  released verifies Hopper async kernels (survey below), so the
  cross-CTA protocols are still covered only by tests and the sanitizer.

## compute-sanitizer sweep (`sweep.sh`)

Suites: `skinny_check.py` (6 shapes incl. split-K, ragged N/K, and the
zero-on-exit workspace contract) and `-k` subsets of
`test_tiered_decode_sliced.py` (4 tests), `test_tiered_decode_moe.py` (6) and
`test_tiered_prefill_moe.py` (10). Each run filters to our namespaces
(`skinny::`, `tiered_decode::`, `tiered_prefill::`; they cover all 16
`__global__` kernels). Results are in `logs/sweep/summary.txt`.

| tool | skinny | sliced | decode | prefill |
|---|---|---|---|---|
| racecheck | 0 hazards | 0 | 0 | 0 |
| synccheck | 0 errors | 0 | 0 | 0 |
| memcheck | 0 errors | 0 | 0 | 0 |
| initcheck (unfiltered) | 0 errors | 0 in ours (5,504 in reference) | 0 in ours (19,456 in reference) | 0 errors |

initcheck must run **without** `--kernel-name`. With a filter it doesn't track
writes from other kernels, so every torch-produced buffer reads as
uninitialized. The filtered skinny run reported 18 "errors": host `.item()`
copies of torch reductions, 6 cases × 3 scalars
(`logs/sweep/skinny.initcheck-filtered.log`). Unfiltered, it reports 0.
Rerun with no print limit, every unfiltered decode-suite error is a 16-byte read
in `vllm::act_and_mul_kernel`. It comes from the tests' `fused_marlin_moe`
reference: `intermediate_cache13` is `torch.empty`, and the activation reads
rows Marlin never wrote. That is upstream reference code, and none of the
errors are in our kernels.

### Positive controls (`mutants/`)

`mutants/skinny_mut.cu` is skinny_gemm with one bug planted per `-DMUT`.
`mutants/run_mut.py <n>` runs K=4096/grid 1 (the 8-stage ring wraps) and
K=512/grid 3 (split-K). Logs: `logs/mut*.log`.

| mutant | output | racecheck | synccheck |
|---|---|---|---|
| 0 baseline | ok (2.4e-3) | 0 | 0 |
| 1 first stage skips its `full` wait | wrong (0.17 / 0.38) | **0** | 1 "Missing wait" |
| 2 consumers arrive on `empty` before reading | wrong at K=4096 (0.54) | **0** | **0** |
| 3 no `bar.sync` between writing `last` and reading it | wrong at grid 3 (1e38) | 1 hazard | 0 |

racecheck tracks only generic-proxy shared accesses: TMA writes into shared
memory are invisible to it. MUT2 is exactly the WAR hazard a TMA ring can have,
and neither tool reports it.

## SentryWard on skinny_gemm (`sentryward/`)

SentryWard ([gpu-verifier/sentryward](https://github.com/gpu-verifier/sentryward))
builds an event graph from PTX at one concrete launch. It checks
well-synchronization (mbarrier phase agreement, no early completion) and
happens-before races. Build: Z3 4.15.1 + Boost 1.88 modules
(`sentryward/swenv.sh`), at `/e/fscratch/profound/naeimitabiei1/tools/sentryward`.

`sentryward/run.sh` reproduces everything below (CPU only, 2m45s). It compiles
`mutants/skinny_mut.cu` at MUT=0..3 to `compute_90a` PTX and prepares
`skinny_kernel<64,128>` with `ptx_prep.py`. Configs come from `mk_skinny.py`
(TMA descriptors for x/w with 128B swizzle, the `Params` struct by offset);
saved copies are in `sentryward/configs/` and `sentryward/reports/`.

| PTX × launch | result | sync | memory |
|---|---|---|---|
| baseline, grid 2, K=512 (no ring wrap) | race_free | passed | passed |
| baseline, grid 1, K=4096 (ring wraps 8× per tile, 80 s) | race_free | passed | passed |
| MUT1, grid 2, K=512 | potential_race (`conflicting_accesses_unordered`) | passed | failed |
| MUT1, grid 1, K=4096 | not_wellsynced (`missing_successful_wait_handoff`) | failed | – |
| MUT2, grid 2, K=512 | race_free: correct, since with no wrap the early release is harmless | passed | passed |
| **MUT2, grid 1, K=4096** | **not_wellsynced (`issue_not_ordered_before_next_phase`)** | failed | – |
| baseline, grid 3 split-K, `last` branch pinned | unsupported (`atomic_add_memory_semantics_unsupported` ×5) | incomplete | – |

Verdicts hold only for the launch analyzed (SentryWard executes it
concretely). MUT2 needs a config where the ring wraps before it can be seen.

### Workarounds needed (`ptx_prep.py`)

1. **`prefetch.tensormap` clobbers its operand register** in SentryWard's
   model, so a later TMA reusing that register lost its descriptor
   (`missing_tma_descriptor`). The prefetch is a cache hint, so prep drops it.
   This is a SentryWard bug and should be reported upstream.
2. **`bfi` is not in the concrete interpreter.** One `bfi.b32` in the
   ldmatrix row address made all 8 ldmatrix addresses unknown
   (`matrix_memory_row_address`). Prep lowers `bfi` with an immediate
   position/length into an equivalent shl/and/or.
3. **`and` on a pointer is not followed.** The `(raw + 1023) & -1024`
   realignment of dynamic shared memory becomes `raw + 0`. This is exact here:
   `raw` is declared `.align 1024`, and SentryWard places the dynamic region
   1024-aligned.

`sentryward/local-debug-prints.patch` holds the two debug prints used to find
these. They are env-gated or stderr-only, and analysis is unchanged.

### What SentryWard cannot reach today

- `tiered_prefill.cu`: `wgmma` is marked unmodeled (`semantics.cpp`).
- skinny split-K and the last-CTA flush, `sliced_decode.cu`'s work-stealing
  `atomicAdd` and ready/last-done flags, `tiered_decode.cu`: global atomics are
  unsupported, and the cross-CTA branches would each need a pinned
  `branch_decisions` entry, with conclusions conditional on that choice.

## Survey: verifiers for Hopper async kernels

None is released that soundly covers TMA + mbarrier + wgmma + proxy fences.

| tool | covers | Hopper async |
|---|---|---|
| SentryWard | PTX event graph: mbarrier phases, TMA bytes, tcgen05/TMEM, races, Folds equivalence | TMA/mbarrier yes (shown here); no wgmma, atomics, data-dependent control |
| Gimlet Labs (blog) | formal equivalence of AI-generated kernels against a reference | not async-pipeline sync |
| VOLTA (OOPSLA'26, [willtunnels/volta](https://github.com/willtunnels/volta)) | symbolic per-block PTX: races, deadlock, equivalence | README: `arrive`/`wgmma` unsupported |
| "AI as a Compiler" ([arXiv 2609.36800](https://arxiv.org/html/2609.36800v1)) | VOLTA plus wgmma/TMA/ldmatrix/tcgen05 | "not sound or complete" (its words); no code |
| HGRD ([arXiv 2604.02106](https://arxiv.org/html/2604.02106)) | static races incl. acquire/release and cross-block | no async; unreleased |
| Faial / FaialAA | race freedom (shared, `__syncthreads`) | no |
| GPUMC ([arXiv 2505.20207](https://arxiv.org/pdf/2505.20207)) | model checking under a scoped weak memory model | no async proxy |
| Dartagnan + PTX 7.5 model | bounded model checking of litmus-size code | no async proxy in the model |
| NVlabs/mixedproxy | Alloy proxy memory model | predates the async proxy |
| WEFT | named-barrier warp specialization | predates mbarrier |
| AccelSync ([arXiv 2605.07881](https://arxiv.org/pdf/2605.07881)) | barrier ordering of cross-unit buffer access | Ascend/Cambricon |
| compute-sanitizer | runtime race/sync/memory checks | blind to TMA hazards (MUT2 above) |

## Next (login-node feasible)

- Pin SentryWard regressions in skinny_gemm: rerun `sentryward/run.sh` after
  any change to its pipeline (CPU only, about 3 min).
- Write the cross-CTA protocols (`sliced_decode` claim/ready/last-done, skinny
  last-CTA flush) as Dartagnan litmus tests against the PTX model. That covers
  acquire/release and scopes, though not the async proxy.
- Report the `prefetch.tensormap` and `bfi` gaps to SentryWard upstream (needs
  the user's go-ahead, since it is outward-facing).
