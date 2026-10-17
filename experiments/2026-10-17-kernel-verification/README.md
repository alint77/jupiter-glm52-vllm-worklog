# Kernel verification on the login node (2026-10-17)

Booster is unavailable (cluster full), so this phase uses only the login-node
GH200 GPU and CPU. The question: what can check our Hopper kernels' sync
(TMA + mbarrier rings, named barriers, split-K atomics, cross-CTA flags)
beyond the correctness tests? Kernels: `skinny_gemm.cu`, `sliced_decode.cu`,
`tiered_decode.cu`, `tiered_prefill.cu`. Two Astra (gpt-6-astra) reviews are
in `astra/`. Their adopted corrections are folded in below; see
[Astra reviews](#astra-reviews).

## Result

- **compute-sanitizer** racecheck, synccheck and memcheck reported no
  diagnostics in the selected, completed runs of all four kernels' correctness
  subsets. In the positive controls, racecheck and synccheck reported nothing
  for MUT2, a consumer that releases its stage before reading it. In the same
  instrumented runs MUT2 produced wrong output, so the sanitizer is no evidence
  of TMA-ring correctness.
- **SentryWard** 60a1ebc reports `race_free` (sync and memory passed) for
  prepared PTX of `skinny_kernel<64,128>` at 7 launches. These cover 4 to 64
  ring iterations and include S−1, S, S+1, 2S+1 and 3S+1 for the 8-stage
  ring. This holds under its modeling assumptions: immediate visibility, no
  fence or proxy semantics, one concrete launch each.
  - It rejects every planted ring bug (9 mutants: missing or early waits,
    early release, expect_tx ±1024, a missing box, wrong parity on either
    barrier, a wrong arrival count, a wrong stage buffer).
  - This is bounded regression evidence for that config only. It does not
    establish Hopper proxy/fence correctness, PTX→SASS correspondence,
    functional correctness, split-K, or skinny configs 0, 2 and 3 (6, 3 and
    6 stages).
- **Open question raised by the reviews:**
  - skinny_gemm and tiered_decode refill a stage with TMA after generic reads
    of it (ldmatrix / lds), with no `fence.proxy.async.shared::cta`;
  - sliced_decode (`sliced_decode.cu:1275`) and tiered_prefill (`:1285`) do
    have that fence;
  - NVIDIA's TMA pipelining guidance calls for synchronization plus a proxy
    fence before reuse;
  - SentryWard can't decide this, since it has no proxy model.

  The fix is cheap (one fence per stage in the producer lane) but changes
  production kernels, so it is for the user to decide.
- Out of SentryWard's reach today: `wgmma` (tiered_prefill), and global
  atomics, flags and fences (skinny split-K, sliced/tiered decode).
  Extending it to the sliced decode kernel is scoped in
  [`sentryward/EXTENSION.md`](sentryward/EXTENSION.md).

## compute-sanitizer sweep (`sweep.sh`)

Suites:

- `skinny_check.py`: 6 shapes, including split-K, ragged N/K, and the
  zero-on-exit workspace check after the second call;
- `-k` subsets of `test_tiered_decode_sliced.py` (4 tests selected),
  `test_tiered_decode_moe.py` (6) and `test_tiered_prefill_moe.py` (10), all
  passing.

racecheck, synccheck and memcheck are filtered to our namespaces
(`skinny::`, `tiered_decode::`, `tiered_prefill::`). A filter only makes a
kernel eligible; it doesn't show that every kernel ran. Results are in
`logs/sweep/summary.txt` (CUDA 13 compute-sanitizer). `sweep.sh` records exit
codes but doesn't fail on diagnostics; read the summary.

| tool | skinny | sliced | decode | prefill |
|---|---|---|---|---|
| racecheck | 0 hazards | 0 | 0 | 0 |
| synccheck | 0 errors | 0 | 0 | 0 |
| memcheck | 0 errors | 0 | 0 | 0 |
| initcheck (unfiltered) | 0 errors | 0 in ours, 5,504 in `act_and_mul` | 0 in ours, 19,456 in `act_and_mul` | 0 errors |

initcheck under `--kernel-name` reported 18 errors on skinny, all host
`.item()` copies of torch reductions. Unfiltered, it reported 0
(`logs/sweep/skinny.initcheck-filtered.log`). That fits the filter dropping
other kernels' writes, so initcheck runs unfiltered here.

Rerun with no print limit, every unfiltered decode-suite error is a 16-byte
read in `vllm::act_and_mul_kernel`. The likely source is the tests'
`fused_marlin_moe` reference: `intermediate_cache13` is `torch.empty`. That's
plausible but **not established**; nobody has checked which rows are read or
whether they're padding. Until then, treat the reference as unaudited for
those rows. None of the errors are in our kernels.

### Positive controls on hardware (`mutants/`)

`mutants/skinny_mut.cu` is skinny_gemm with one bug planted per `-DMUT`.
`mutants/run_mut.py <n>` runs K=4096/grid 1 (64 iterations, 8 uses per stage)
and K=512/grid 3 (split-K). Logs: `logs/mut*.log`. The output error comes from
the same instrumented invocation.

| mutant | output | racecheck | synccheck |
|---|---|---|---|
| 0 baseline | ok (2.4e-3) | 0 | 0 |
| 1 first stage skips its `full` wait | wrong (0.17 / 0.38) | **0** | 1 "Missing wait" |
| 2 consumers arrive on `empty` before reading | wrong at K=4096 (0.54) | **0** | **0** |
| 3 no `bar.sync` between writing `last` and reading it | wrong at grid 3 (1e38) | 1 hazard | 0 |

That racecheck tracks no TMA writes is a hypothesis these runs fit, not a
demonstrated mechanism. Minimal TMA RAW/WAR/WAW litmus cases would settle it.
MUT4–12 were not run on hardware: several hang by design.

## SentryWard on skinny_gemm (`sentryward/`)

SentryWard ([gpu-verifier/sentryward](https://github.com/gpu-verifier/sentryward))
builds an event graph from PTX at one concrete launch, then checks
well-synchronization (mbarrier phase agreement, no early completion) and
happens-before races. Build: Z3 4.15.1 + Boost 1.88 modules
(`sentryward/swenv.sh`), at `/e/fscratch/profound/naeimitabiei1/tools/sentryward`.

`sentryward/run.sh` does the whole run in 6 min of CPU:

- writes into a fresh run directory under
  `/e/fscratch/profound/naeimitabiei1/sentryward-runs/`, with
  `provenance.txt` (nvcc version, git HEAD, source and PTX hashes);
- compiles each mutant to `compute_90a` PTX and prepares it with
  `ptx_prep.py`;
- generates the configs with `mk_skinny.py`;
- asserts every verdict and exits nonzero on any failure or mismatch.

One stage iteration is one 128-wide K chunk. Grid 2 at N=128 gives each CTA
one tile, so K/128 iterations. Results from run `20261017-024344`:

| case | iterations | result | expected |
|---|---|---|---|
| baseline, grid 2, K = 512 / 896 / 1024 / 1152 / 2176 / 3200 | 4 / 7 / 8 / 9 / 17 / 25 | race_free | ok |
| baseline, grid 1, K=4096 | 64 | race_free | ok |
| MUT1 skip first `full` wait, K=512 | 4 | potential_race | bug |
| MUT2 early `empty` arrive, K=512 | 4 (no reuse) | race_free | ok: no reuse, so no reuse hazard |
| MUT1 / MUT2, K=2176 | 17 | not_wellsynced | bug |
| MUT4 expect_tx −1024 / MUT5 +1024 / MUT6 one X box missing | 17 | not_wellsynced | bug |
| MUT7 wrong `full` parity / MUT8 wrong `empty` parity / MUT9 `empty` count −1 | 17 | not_wellsynced | bug |
| MUT12 producer fills the wrong stage buffer | 17 | potential_race | bug |
| MUT10 no `__syncwarp` before lane 0's release | 17 | potential_race | (model: other lanes' ldmatrix reads unordered) |
| MUT11 no `fence.mbarrier_init` | 17 | race_free | (outside the model: fences are not modeled) |
| baseline, grid 3 split-K, `last` branch pinned | – | unsupported (`atomic_add_memory_semantics_unsupported`) | unsupported |

Not-wellsynced mutants fail in the synchronization check, so their memory
check doesn't run. The MUT2 finding at reuse is a synchronization rejection,
not a direct WAR witness.

### PTX preparation (`ptx_prep.py`)

Each rewrite is validated, and prep fails if a site count differs from the
`--expect-*` value (2 / 3 / 1 for skinny):

1. **Drop `prefetch.tensormap [%rdN];`.** SentryWard's model loses the operand
   register's tensor-map binding, so a later TMA reusing it reported
   `missing_tma_descriptor`. Dropping a cache hint is defensible for sync
   properties: the maps are `__grid_constant__` and never updated. It doesn't
   validate descriptor legality. A minimal upstream reproducer is still to do.
2. **Lower `bfi.b32`** with in-range immediate position/length to shl/and/or.
   The interpreter lacks `bfi`, which made all 8 ldmatrix row addresses
   unknown. Astra checked the arithmetic: mask `0xfffffff0`, complement `15`.
3. **`(p + 1023) & -1024` → `p + 0`**, only for the exact def-use chain from
   a `mov` of a shared symbol declared `.extern .shared .align 1024` through
   `cvta.shared`. The interpreter doesn't follow `and` on a pointer. This is
   exact given the declared alignment of the emitted symbol.

`sentryward/local-debug-prints.patch` holds the two debug prints used to find
these. They are env-gated or stderr-only.

## Astra reviews

Both reviews are in `astra/`.

**`review1.md` (this write-up).** Adopted:

- qualified the proof claims;
- `run.sh` now uses fresh dirs, checks every status, asserts verdicts and
  records provenance;
- fixed the iteration counts (64 iterations at K=4096 grid 1, not "8 wraps
  per tile");
- added the iteration sweep and MUT4–12;
- `ptx_prep.py` validates its sites;
- softened the initcheck and racecheck-mechanism claims.

Not done yet:

- TMA litmus cases for the sanitizer;
- the initcheck row audit;
- configs 0, 2, 3;
- per-call workspace checks in `skinny_check.py`.

**`extension-review.md` (the sliced-kernel extension).** No definite
synchronization bug in `layer_kernel`'s handoffs:

- routed y13 → finisher → x2 → ready → producer;
- the `hq`/`sq` generations;
- stage reuse;
- the `K_END` and steal paths.

The shared-expert `done_s` collection (lines 1503–1512) is ordered only under
full SC-fence semantics of `__threadfence`; an acquire/release approximation
doesn't cover it. Hardening option: an acq_rel `done_s` RMW before the
`bar.sync`. The design corrections are in EXTENSION.md.

## Survey: verifiers for Hopper async kernels

Among the surveyed versions, none covers the full TMA + mbarrier + wgmma +
proxy-fence combination.

| tool | covers | Hopper async |
|---|---|---|
| SentryWard | PTX event graph: mbarrier phases, TMA bytes, tcgen05/TMEM, races, Folds equivalence | TMA/mbarrier yes (shown here); no wgmma, atomics, fences/proxies, data-dependent control |
| Gimlet Labs (blog) | formal equivalence of AI-generated kernels against a reference | not async-pipeline sync |
| VOLTA (OOPSLA'26, [willtunnels/volta](https://github.com/willtunnels/volta)) | symbolic per-block PTX: races, deadlock, equivalence | README: `arrive`/`wgmma` unsupported |
| "AI as a Compiler" ([arXiv 2609.36800](https://arxiv.org/html/2609.36800v1)) | VOLTA plus wgmma/TMA/ldmatrix/tcgen05 | "not sound or complete" (its words); no code |
| HGRD ([arXiv 2604.02106](https://arxiv.org/html/2604.02106)) | static races incl. acquire/release and cross-block | no async; unreleased |
| Faial / FaialAA | race freedom (shared, `__syncthreads`) | no |
| GPUMC ([arXiv 2505.20207](https://arxiv.org/pdf/2505.20207)) | model checking under a scoped weak memory model | no async proxy |
| Dartagnan + PTX model | bounded model checking of litmus-size code | no async proxy in the model; audit per feature |
| NVlabs/mixedproxy | Alloy proxy memory model | research prototype, limited proxy coverage |
| WEFT | named-barrier warp specialization | predates mbarrier |
| AccelSync ([arXiv 2605.07881](https://arxiv.org/pdf/2605.07881)) | barrier ordering of cross-unit buffer access | Ascend/Cambricon |
| compute-sanitizer | runtime race/sync/memory checks | no diagnostics for MUT2 (above) |

## Next (login-node feasible)

- Rerun `sentryward/run.sh` after any change to skinny_gemm's pipeline.
- Settle the stage-reuse proxy-fence question for skinny_gemm and tiered_decode
  (user decision; a production kernel change).
- For the sliced decode kernel, follow EXTENSION.md: hand-extracted litmus
  tests first, then SentryWard at GRID=1.
- Report the `prefetch.tensormap` and `bfi` gaps upstream, with minimal
  reproducers. That needs the user's go-ahead, since it is outward-facing.
