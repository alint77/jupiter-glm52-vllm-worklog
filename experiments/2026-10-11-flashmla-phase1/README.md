# FlashMLA sparse decode Phase 1: balanced schedule, early stop, register split

2026-10-11. This follows the Phase 0 measurements (`../2026-10-11-flashmla-phase0`)
and changes three things, all with the same math:

1. **Balanced split schedule built by vLLM.** For b = 1 (all T decode tokens
   in one request), FlashMLA's planner leaves the splits unbalanced (9 + 3
   blocks at T = 48/64, 5 + 5 + 2 at T = 32).
   `dcp_decode_sched_meta` (`vllm/v1/attention/backends/mla/sparse_utils.py`)
   is a one-CTA Triton kernel that writes the `DecodingSchedMeta` and
   `num_splits` on device:
   - `longest = ceil(nb / parts)`, `splits = ceil(nb / longest)`, bounds
     `nb * i // splits`;
   - idle parts start at request 1 = b, so they exit at once.

   The indexer layers build the schedule together with their converted
   indices. The reuse layers take both from `_DCP_CONVERTED_INDICES` (21
   schedule kernels per step, not 78). The call goes straight to
   `torch.ops._flashmla_C.sparse_decode_fwd` with the schedule, so the
   planner never runs.
2. **Early stop.** `nb = ceil(min(max(valid_counts), W) / 64)`. DCP index rows
   hold their live slots as a prefix (compacted by
   `triton_filter_and_convert_dcp_index`, -1 after), so blocks past the
   longest live row are -1 in every row. `valid_counts` stays exact for rows
   that overflow the width, hence the clamp to W.
3. **setmaxnreg 208/160/136** (WG0 / WG1 / producer; stock 192/160/152) for
   the V3.2 h64 kernel. The change is on FlashMLA branch `dcp-decode-regs`
   (`e2f70d4`, on vLLM's pin `a8f794d`; worktree
   `src/FlashMLA-regs`, build `build/fmla-regs`).
   - It removes ptxas C7512 (all 44 HGMMAs serialized): 3 DEPBARs instead of
     one per HGMMA.
   - The sum must be ≤ 3 x 168 = 504 (`static_assert`, plus a host
     `KU_ASSERT` against `cudaFuncGetAttributes().numRegs`).
   - Only the V3.2 h64 sparse kernel's SASS differs from prod.

Env `VLLM_DCP_DECODE_SCHED` (default 1; 0 restores the planner). The schedule
is used only on sm90 without head padding. vLLM commit `80bd3cce5c`.

## Correctness

**Register build: bitwise vs prod** (`regs_hashes.py`, `run_tests.sh
hashes`). The script hashes out and lse for 3168 configs:
- every T in 1..64, plus 100 and 140;
- widths 64 / 576 / 768 / 2048;
- count patterns live, boundary, single key, empty, overflow and full;
- both the planner and the balanced schedule.

Each library runs in its own process. Prod (`3c6f31e7`) vs the stock rebuild
(`b5843aa9`): 0 differ. Prod vs the register build (`9c2356f2`): **0 differ**.

**vLLM tests** (`tests/v1/attention/test_flashmla_sparse_dcp.py`; `run_tests.sh
tests`). 133 pass with the register library (worktree `fmla-p1`) and with the
prod library (`fmla-p1-stock`). New tests:

- **Schedule invariants** for parts in {1, 2, 4, 5, 8, 16, 33, 66, 132} and
  widths 64..2048. Count patterns: empty, 1, 63/64/65, W-1, W, past W (rows
  that overflowed), the longest row after 299 empty ones (the second
  256-token chunk), and random. Each case checks:
  - contiguous coverage of `[0, nb)`;
  - no empty split;
  - longest split = `ceil(nb/parts)`, in the fewest splits;
  - idle parts at request 1;
  - `num_splits`.
- **Scheduled vs planner vs fp32 reference** for T in {1, 2, 7, 16, 33, 64,
  140} x W in {64, 768, 2048} x patterns live / boundary / single key /
  empty / overflow.
  - Data: q x2 with every other token's q = 0 (uniform attention), zero
    groups, one group 200x the others, repeated indices, and the last cache
    row.
  - Empty rows keep a non-finite LSE and a finite output.
  - Outputs agree within `1e-2 * (P @ |V|) + 1e-3` per element (bf16 P and
    output rounding bound). LSE is within 1e-4 of the planner and 2e-3 of the
    reference.
- **4-rank DCP combine** on the scheduled path (T = 1, 4, 33), with one row
  whose keys all live on rank 0.
- **CUDA graph** capture of schedule plus attention, replayed with counts
  1 → W → ~500 → one row 65 → all empty → overflow → 64.
- **Reuse-layer cache**:
  - the same tensors are shared;
  - a lazily built schedule is written back once;
  - a layer with 128 padded heads gets its own part count;
  - prefill never gets a schedule.
- **Skip-KV staging**: the scheduled path on the staged buffer is
  bit-identical to the store (staging keeps each live slot in its column).

The fp32 reference has to dequantize as the kernel does. A probe with one live
key per row (`out = V`) showed the kernel computes `bf16(scale) * fp8`,
rounded to bf16. This matches the kernel 100% bit-exactly. The fp32-scale
variants were off by up to 8 (scores up to ~500 in the 200x group).

**Mutation check** (`mutant.sh`). A schedule that drops the last partial
block (`live // 64`) fails 26 tests: the invariants, the serving-like output
cases, graph replay and reuse.

## Astra review (gpt-6-astra, prompt `astra-p1.md` inlined)

It found no arithmetic counterexample:
- `1 ≤ S ≤ min(nb, P)`;
- the bounds give contiguous coverage with no empty split;
- `begin_split_idx = part` with `num_splits[0] = 0` is right;
- idle parts exit;
- single-split empty rows write +inf, like the combine;
- 208/160/136 fits the 504 budget (WG0 +40, WG1 -8, producer -32).

Findings and what was done:

| finding | action |
| --- | --- |
| P1 (conditional): staging replaces indices after the schedule is chosen; this needs live keys to stay in their columns | staging remaps elementwise and keeps -1 positions; added a scheduled-path bitwise check to the staging test |
| P2: a cached schedule depends on SM count and padded heads; an h128 reuse layer would need other parts | reuse checks the part count and rebuilds; test added |
| P2: tests bypass the production forward and streams | served A/B below (real forward, CUDA graphs, side-stream indexer); the schedule is produced and consumed on the attention stream, after the existing indexer join |
| P3: lazily built schedule not written back | written back; test asserts one build |
| P3: `static num_regs` is not per device | homogeneous GH200; noted |
| tests: uniform attention, 4-rank combine | added (q = 0 rows; scheduled shards test) |

## Microbenchmark (`bench_integrated.py`, graph replay of one 78-layer step)

prod: planner once per step, plus `flash_mla_with_kvcache` per layer. new:
`dcp_decode_sched_meta` on 21 indexer layers, plus the direct call. W = 768
with 470-540 live keys. GPU 0 used the register library and GPU 1 the prod
library (`logs/bench-*.log`). All values are µs per step.

| T | prod lib, planner (prod) | prod lib, new schedule | regs lib, planner | regs lib, new schedule | total saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1072.0 | 1031.0 | 969.0 | 928.9 | **-143** |
| 8 | 1288.1 | 1170.8 | 1174.6 | 1081.9 | **-206** |
| 16 | 1577.3 | 1395.1 | 1473.1 | 1302.0 | **-275** |
| 32 | 2134.2 | 1872.3 | 2037.1 | 1764.2 | **-370** |
| 64 | 3353.9 | 2455.8 | 3284.4 | 2326.9 | **-1027** |

These include the schedule kernels. Max |diff| vs the planner is 4.9e-4 (1 bf16 ulp).
The library columns come from different GPUs, so the register share carries
cross-GPU noise. It is in line with Phase 0's 0.7-2.1 µs per layer.

## Served A/B

Same-node A/B on the MTP3 TP-sliced c16 config (`p1_arm.sh`, 400K pool per
request, captures to 64 tokens). base = `dcp-new` (4524b36e4e, prod library);
new = `fmla-p1` (80bd3cce5c plus the register library). Each node ran both arms
in opposite order: `ab_launch.sh` (2, 8 and 16 requests plus GSM8K; holds
2271866 / 2271867) and `ab_launch_ns.sh` (1, 4 and 12 requests; holds 2271974 /
2271975). `ab_table.py` prints the ms/step deltas below (new - base, per node).

| context | requests | base ms | new ms | per node (new - base) | mean |
| --- | ---: | ---: | ---: | --- | ---: |
| 5K | 1 | 17.54 | 17.25 | -0.23 -0.36 | **-0.29** (-1.7%) |
| 50K | 1 | 17.76 | 17.77 | +0.28 -0.25 | **+0.01** (+0.1%) |
| 5K | 2 | 21.17 | 20.82 | -0.42 -0.29 | **-0.36** (-1.7%) |
| 50K | 2 | 21.64 | 21.10 | -0.56 -0.51 | **-0.54** (-2.5%) |
| 5K | 4 | 27.64 | 27.40 | -0.28 -0.18 | **-0.23** (-0.8%) |
| 50K | 4 | 28.28 | 28.33 | +0.07 +0.02 | **+0.05** (+0.2%) |
| 5K | 8 | 38.84 | 38.34 | -0.77 -0.23 | **-0.50** (-1.3%) |
| 50K | 8 | 40.76 | 40.59 | +0.28 -0.64 | **-0.18** (-0.4%) |
| 5K | 12 | 47.98 | 47.10 | -1.48 -0.28 | **-0.88** (-1.8%) |
| 50K | 12 | 50.16 | 49.21 | -1.78 -0.12 | **-0.95** (-1.9%) |
| 5K | 16 | 57.32 | 56.31 | -0.38 -1.62 | **-1.00** (-1.8%) |
| 50K | 16 | 60.05 | 58.36 | -1.21 -2.16 | **-1.69** (-2.8%) |

- 20 of 24 node pairs favour new.
- The gain grows with tokens per step, as in the microbenchmark: about -0.3 ms
  at 1-4 requests (with two 50K cells inside noise) and -1 to -1.7 ms at 16.
- Acceptance is unchanged (2.75-3.53 tokens per step on both arms).
- GSM8K: base 0.925 / 0.92, new 0.91 / 0.92 (within noise: the register build
  is bitwise identical, and the schedule moves split bounds only, 1 bf16 ulp).

## Ship

- `serve.sh` (`2026-09-27-glm53-mtp7-profile`) exports
  `VLLM_DCP_DECODE_SCHED=1`.
- The register library is installed as `vllm/_flashmla_C.abi3.so` in the main
  tree (md5 `9c2356f2`; the prod backup `3c6f31e7` is in
  `build/fmla-prod-backup`).
- Prod (`claude-glm53-sliced-df2-dcp4.sh`, job 2272095) restarted on it with
  3.41-3.43 GiB observed free.
- The c2/c4/c8/c16 MTP3 launchers (vLLM `244d3997a5`) booted on it.
