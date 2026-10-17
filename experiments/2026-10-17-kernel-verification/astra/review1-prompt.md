# Review request: GPU kernel verification results (Hopper / GH200, sm_90a)

You are reviewing an experiment, not running it. **Do not run any commands** (your sandbox cannot start on this machine); reason from the material below. Be adversarial: look for unsound steps, overclaims, wrong interpretations, missing controls. Cite the section/line you object to.

Context: we wrote custom CUDA kernels for vLLM MoE decode on GH200 (TMA + mbarrier rings, warp specialization, split-K atomics, cross-CTA flags). We checked them with (1) NVIDIA compute-sanitizer and (2) SentryWard (github.com/gpu-verifier/sentryward, commit 60a1ebc), a PTX verifier built for Blackwell that builds an event graph from one concrete launch and checks mbarrier well-synchronization + happens-before races. SentryWard assumptions (from its docs): immediate visibility, no fences modeled, no data-dependent control flow (branches must be static or pinned via `branch_decisions`, which makes conclusions conditional).

Questions:
1. Are the three PTX-prep rewrites (drop `prefetch.tensormap`, lower `bfi`, `(p+1023)&-1024 -> p+0`) sound, i.e. does verifying the rewritten PTX say anything about the real kernel? Check the bfi lowering arithmetic.
2. Is "proves skinny_gemm's in-CTA TMA/mbarrier pipeline race-free and well-synchronized at two launches" justified? What does SentryWard's guarantee NOT cover here (e.g. proxy ordering generic-read -> async-proxy TMA write; parity aliasing; expect_tx byte counts; the PTX->SASS gap; other launch shapes / configs 0,2,3)?
3. Interpretation of the mutants: MUT2 race_free at K=512 grid 2 claimed "correct, no ring wrap". Right? Is MUT1 being potential_race at one config and not_wellsynced at the other plausible?
4. compute-sanitizer claims: "racecheck blind to TMA writes", initcheck filter artifact, act_and_mul errors attributed to the Marlin reference's torch.empty cache. Anything wrong or under-supported?
5. What further mutants/controls would most strengthen or falsify the conclusions (e.g. wrong expect_tx bytes, wrong parity, STAGES off-by-one, missing fence.mbarrier_init)?
6. Anything in the README that overclaims and should be reworded.

Give findings ranked by severity, each with: claim, why it is wrong/weak, concrete fix or extra check.

## README.md (the write-up under review)
```markdown
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
```

## sentryward/ptx_prep.py
```python
"""Prepare one PTX entry for SentryWard.

Keeps the header and one .entry, drops `prefetch.tensormap` (a cache hint;
SentryWard 60a1ebc clobbers its operand register, losing the map pointer
the later TMA reuses), and with --assume-aligned rewrites the
`(p + 1023) & -1024` dynamic-smem realignment to `p + 0`, i.e. assumes the
dynamic shared base is already 1024-aligned (SentryWard cannot follow `and`
on a pointer). `bfi` with immediate position/length is lowered to
and/shl/or, which SentryWard's concrete interpreter evaluates.
"""
import re
import sys

src, entry, dst = sys.argv[1:4]
aligned = "--assume-aligned" in sys.argv
lines = open(src).read().split("\n")
starts = [i for i, l in enumerate(lines) if ".entry " in l]
first = starts[0]
k = next(i for i in starts if entry in lines[i])
end = next((i for i in starts if i > k), len(lines))
out = lines[:first] + lines[k:end]
out = [l for l in out if "prefetch.tensormap" not in l]
if aligned:
    pend = {}
    for i, l in enumerate(out):
        m = re.match(r"\s*add\.s64\s+(%rd\d+), (%rd\d+), 1023;", l)
        if m:
            pend[m.group(1)] = (i, m.group(2))
        m = re.match(r"\s*and\.b64\s+(%rd\d+), (%rd\d+), -1024;", l)
        if m and m.group(2) in pend:
            j, base = pend[m.group(2)]
            out[i] = f"\tadd.s64 \t{m.group(1)}, {base}, 0;"
bfi = re.compile(r"(\s*)bfi\.b32\s+(%r\d+), (%r\d+), (%r\d+), (\d+), (\d+);")
lowered = []
for l in out:
    m = bfi.match(l)
    if not m:
        lowered.append(l)
        continue
    ind, d, a, b, pos, n = m.groups()
    mask = ((1 << int(n)) - 1) << int(pos) & 0xFFFFFFFF
    lowered += [
        f"{ind}{{ .reg .b32 %bfa, %bfb;",
        f"{ind}shl.b32 %bfa, {a}, {pos};",
        f"{ind}and.b32 %bfa, %bfa, {mask};",
        f"{ind}and.b32 %bfb, {b}, {~mask & 0xFFFFFFFF};",
        f"{ind}or.b32 {d}, %bfa, %bfb; }}",
    ]
out = lowered
open(dst, "w").write("\n".join(out))
```

## sentryward/mk_skinny.py (SentryWard launch config)
```python
"""SentryWard input for skinny_kernel<64,128> (4 consumer warps + 1 TMA
producer, 8 stages). Usage: mk_skinny.py <grid> <K> <out.json>."""
import json, sys

M, N, TN, CK = 8, 128, 64, 128
grid, K = int(sys.argv[1]), int(sys.argv[2])
tiles, chunks = (N + TN - 1) // TN, (K + CK - 1) // CK
cfg = {
  "schema": "spverify.kernel-input/v1",
  "entry": "_ZN6skinny13skinny_kernelILi64ELi128EEEv14CUtensorMap_stS1_NS_6ParamsE",
  "launch": {"grid_dim": [grid, 1, 1], "block_dim": [160, 1, 1],
             "cluster_dim": [1, 1, 1], "dynamic_shared_bytes": 1024 + 8 * (TN * CK * 2 + 8 * CK * 2)},
  "declarations": [
    {"parameter": 2, "offset": 24, "type": "pointer", "space": "global", "object": "out"},
    {"parameter": 2, "offset": 32, "type": "pointer", "space": "global", "object": "ws"},
    {"parameter": 2, "offset": 40, "type": "pointer", "space": "global", "object": "counters"},
  ],
  "arguments": [{"parameter": 2, "offset": o, "type": "s32", "value": v}
                for o, v in [(0, M), (4, N), (8, K), (12, tiles), (16, chunks)]],
  "tma_descriptors": [
    {"parameter": 0, "offset": 0, "object": "w", "element_type": "bf16", "rank": 2,
     "global_shape": [K, N], "coordinate_strides_bytes": [2, 2 * K], "box_shape": [64, TN],
     "element_strides": [1, 1], "swizzle": "128B", "interleave": "none", "oob_fill": "zero"},
    {"parameter": 1, "offset": 0, "object": "x", "element_type": "bf16", "rank": 2,
     "global_shape": [K, M], "coordinate_strides_bytes": [2, 2 * K], "box_shape": [64, 8],
     "element_strides": [1, 1], "swizzle": "128B", "interleave": "none", "oob_fill": "zero"},
  ],
}
json.dump(cfg, open(sys.argv[3], "w"), indent=1)
```

## sentryward/run.sh
```bash
#!/bin/bash
# SentryWard 60a1ebc on skinny_kernel<64,128> sm_90a PTX: baseline + mutants.
# Usage: sentryward/run.sh [workdir]   (CPU only; ~3 min)
set -u
D=$(cd "$(dirname "$0")" && pwd)
E=$(dirname "$D")
R=$(cd "$E/../../.." && pwd)
W=${1:-/e/fscratch/profound/naeimitabiei1/sentryward-runs}
mkdir -p "$W"
source "$D/swenv.sh"
P=$R/.venv/bin/python
INC=$($P -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths('cuda')))")
PYI=$($P -c "import sysconfig; print(sysconfig.get_paths()['include'])")
ENTRY=skinny_kernelILi64ELi128

for m in 0 1 2 3; do
  nvcc -ptx -O3 -std=c++17 -arch=compute_90a $INC -I"$PYI" \
    -DTORCH_EXTENSION_NAME=skinny -DMUT=$m -I"$R/vllm/model_executor/layers/skinny_gemm" \
    "$E/mutants/skinny_mut.cu" -o "$W/mut$m.ptx" 2>/dev/null
  $P -I "$D/ptx_prep.py" "$W/mut$m.ptx" $ENTRY "$W/mut${m}_al.ptx" --assume-aligned
done
# K=512 grid 2: one tile per CTA, 8 K-steps = no ring wrap.
# K=4096 grid 1: two tiles x 64 K-steps, the 8-stage ring wraps 8x per tile.
$P -I "$D/mk_skinny.py" 2 512 "$W/g2_k512.json"
$P -I "$D/mk_skinny.py" 1 4096 "$W/g1_k4096.json"

run() {  # name ptx config
  timeout 2400 "$SWB" --ptx "$2" --config "$3" --report "$W/$1.json" >/dev/null 2>&1
  $P -I -c "
import collections, json, sys
r = json.load(open(sys.argv[1]))
codes = collections.Counter(d.get('code') for d in r.get('diagnostics', []))
print(f'{sys.argv[2]:<16} {r[\"result\"]:<16}', r['checks']['synchronization'],
      r['checks']['memory'], dict(codes) or '')" "$W/$1.json" "$1"
}
for m in 0 1 2; do
  run mut${m}_g2_k512 "$W/mut${m}_al.ptx" "$W/g2_k512.json"
  run mut${m}_g1_k4096 "$W/mut${m}_al.ptx" "$W/g1_k4096.json"
done
# Split-K (grid 3): pin the cross-CTA last-arriver branch to "flush".
L=$(awk '/ld.shared.u32.*4last\]/{f=1} f && /bra/{print NR; exit}' "$W/mut0_al.ptx")
$P -I -c "
import json, sys
c = json.load(open(sys.argv[1])); c['launch']['grid_dim'] = [3, 1, 1]
c['branch_decisions'] = [{'line': int(sys.argv[2]), 'taken': False,
                          'note': 'cross-CTA last-arriver flag: this CTA flushes'}]
json.dump(c, open(sys.argv[3], 'w'), indent=1)" "$W/g2_k512.json" "$L" "$W/g3_k512_last.json"
run mut0_g3_splitk "$W/mut0_al.ptx" "$W/g3_k512_last.json"
```

## run.sh output (verbatim)
```
mut0_g2_k512     race_free        passed passed
mut0_g1_k4096    race_free        passed passed
mut1_g2_k512     potential_race   passed failed {'conflicting_accesses_unordered': 1}
mut1_g1_k4096    not_wellsynced   failed not_run {'missing_successful_wait_handoff': 1}
mut2_g2_k512     race_free        passed passed
mut2_g1_k4096    not_wellsynced   failed not_run {'issue_not_ordered_before_next_phase': 1}
mut0_g3_splitk   unsupported      incomplete not_run {'atomic_add_memory_semantics_unsupported': 5}
```

## skinny_gemm.cu (production kernel, full, with line numbers)
```cuda
     1	// SPDX-License-Identifier: Apache-2.0
     2	// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
     3	//
     4	// y[M, N] = x[M, K] @ W[N, K]^T for bf16 at decode sizes (M <= 8) on sm_90a.
     5	//
     6	// At a handful of tokens a linear layer is a weight read: the job is to stream
     7	// W at HBM speed on every SM. Library GEMMs at M = 8 pick 64x8 tiles, split-K
     8	// with a separate reduce kernel, or under-fill the GPU for small N; this
     9	// kernel spreads (128-row tile, 128-wide K chunk) units evenly over all SMs
    10	// (stream-K). A producer warp feeds a TMA ring (128-byte swizzle, so the
    11	// ldmatrix reads below are bank-conflict free); eight consumer warps each own
    12	// 16 rows of the tile and run mma.m16n8k16 with W as A and the tokens as B.
    13	//
    14	// A CTA that owns a whole tile writes bf16 directly. Otherwise it adds its
    15	// fp32 partial into a workspace and counts its chunks into the tile's
    16	// counter; the CTA that completes the count converts the tile, zeroes the
    17	// workspace rows and resets the counter, so every call finds them clean and
    18	// there is no second launch.
    19	
    20	#include <cuda.h>
    21	#include <cuda_bf16.h>
    22	#include <cuda_runtime.h>
    23	#include <torch/extension.h>
    24	#include <c10/cuda/CUDAStream.h>
    25	
    26	#include <algorithm>
    27	#include <cstdint>
    28	
    29	namespace skinny {
    30	
    31	constexpr int HALF_K = 64;  // one 128-byte swizzled box width, in bf16
    32	constexpr int MAX_M = 8;
    33	constexpr int GRID = 132;
    34	constexpr int SMEM_LIMIT = 227 * 1024;
    35	
    36	// TILE_N rows per tile (16 per consumer warp), CHUNK_K columns per unit
    37	template <int TILE_N, int CHUNK_K>
    38	struct Cfg {
    39	  static constexpr int CONSUMERS = TILE_N / 16;
    40	  static constexpr int THREADS = (CONSUMERS + 1) * 32;
    41	  static constexpr int BOXES = CHUNK_K / HALF_K;
    42	  static constexpr int W_BYTES = TILE_N * CHUNK_K * 2;
    43	  static constexpr int X_BYTES = MAX_M * CHUNK_K * 2;
    44	  static constexpr int STAGE_BYTES = W_BYTES + X_BYTES;
    45	  static constexpr int STAGES = std::min(8, (SMEM_LIMIT - 2048) / STAGE_BYTES);
    46	  static constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES;  // 1 KB swizzle align
    47	};
    48	
    49	__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    50	  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
    51	}
    52	__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    53	  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
    54	}
    55	__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    56	  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
    57	               "r"(bytes) : "memory");
    58	}
    59	__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    60	  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
    61	}
    62	__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    63	  asm volatile(
    64	      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
    65	      "p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(parity)
    66	      : "memory");
    67	}
    68	__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, int x, int y,
    69	                                       uint64_t* bar) {
    70	  asm volatile(
    71	      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    72	      " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
    73	      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
    74	      : "memory");
    75	}
    76	__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
    77	  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    78	               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
    79	}
    80	__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, uint32_t addr) {
    81	  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
    82	               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
    83	}
    84	__device__ __forceinline__ void mma(float* d, const uint32_t* a, const uint32_t* b) {
    85	  asm volatile(
    86	      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    87	      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    88	      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    89	      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    90	}
    91	// Byte offset of (row, 16-byte column c) in a 128-byte-wide box written with
    92	// CU_TENSOR_MAP_SWIZZLE_128B: the column is XORed with the row within 8.
    93	__device__ __forceinline__ uint32_t swz(int row, int c) {
    94	  return row * 128 + ((c ^ (row & 7)) << 4);
    95	}
    96	
    97	struct Params {
    98	  int M, N, K, tiles, chunks;
    99	  __nv_bfloat16* out;
   100	  float* ws;      // [MAX_M, N] fp32, zero between calls
   101	  int* counters;  // [tiles], zero between calls
   102	};
   103	
   104	template <int TILE_N, int CHUNK_K>
   105	__global__ void __launch_bounds__(Cfg<TILE_N, CHUNK_K>::THREADS, 1)
   106	    skinny_kernel(const __grid_constant__ CUtensorMap wmap,
   107	                  const __grid_constant__ CUtensorMap xmap, const Params p) {
   108	  using C = Cfg<TILE_N, CHUNK_K>;
   109	  constexpr int STAGES = C::STAGES, CONSUMERS = C::CONSUMERS, BOXES = C::BOXES;
   110	  constexpr int W_BYTES = C::W_BYTES, X_BYTES = C::X_BYTES, STAGE_BYTES = C::STAGE_BYTES;
   111	  extern __shared__ __align__(1024) unsigned char raw[];
   112	  unsigned char* ring = reinterpret_cast<unsigned char*>(
   113	      (reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
   114	  __shared__ uint64_t full[STAGES], empty[STAGES];
   115	  __shared__ int last;
   116	  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
   117	  const int64_t units = static_cast<int64_t>(p.tiles) * p.chunks;
   118	  const int u0 = static_cast<int>(units * blockIdx.x / gridDim.x);
   119	  const int u1 = static_cast<int>(units * (blockIdx.x + 1) / gridDim.x);
   120	
   121	  if (threadIdx.x == 0) {
   122	    for (int s = 0; s < STAGES; ++s) {
   123	      mbar_init(&full[s], 1);
   124	      mbar_init(&empty[s], CONSUMERS);
   125	    }
   126	    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
   127	  }
   128	  __syncthreads();
   129	  if (u0 >= u1) return;
   130	
   131	  if (warp == CONSUMERS) {  // producer
   132	    if (lane == 0) {
   133	      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&wmap))
   134	                   : "memory");
   135	      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&xmap))
   136	                   : "memory");
   137	      for (int u = u0, it = 0; u < u1; ++u, ++it) {
   138	        const int t = u / p.chunks, c = u - t * p.chunks, s = it % STAGES;
   139	        if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
   140	        unsigned char* st = ring + s * STAGE_BYTES;
   141	        mbar_expect_tx(&full[s], STAGE_BYTES);
   142	        for (int h = 0; h < BOXES; ++h) {
   143	          const int k = c * CHUNK_K + h * HALF_K;
   144	          tma_2d(st + h * (W_BYTES / BOXES), &wmap, k, t * TILE_N, &full[s]);
   145	          tma_2d(st + W_BYTES + h * (X_BYTES / BOXES), &xmap, k, 0, &full[s]);
   146	        }
   147	      }
   148	    }
   149	    return;
   150	  }
   151	
   152	  // consumer: rows [warp*16, warp*16+16) of the tile
   153	  const int g = lane / 4, tq = lane % 4;
   154	  float acc[4] = {0.f, 0.f, 0.f, 0.f};
   155	  int first_chunk = -1;
   156	  for (int u = u0, it = 0; u < u1; ++u, ++it) {
   157	    const int t = u / p.chunks, c = u - t * p.chunks, s = it % STAGES;
   158	    if (first_chunk < 0) first_chunk = c;
   159	    mbar_wait(&full[s], (it / STAGES) & 1);
   160	    const uint32_t w = smem_u32(ring + s * STAGE_BYTES);
   161	    const uint32_t x = w + W_BYTES;
   162	#pragma unroll
   163	    for (int h = 0; h < BOXES; ++h) {
   164	#pragma unroll
   165	      for (int kk = 0; kk < HALF_K / 16; ++kk) {
   166	        // A: rows warp*16 + (lane%16), 16-byte column 2kk + lane/16
   167	        uint32_t a[4], b[2];
   168	        const int arow = warp * 16 + (lane & 15);
   169	        ldmatrix_x4(a, w + h * (W_BYTES / BOXES) + swz(arow, 2 * kk + (lane >> 4)));
   170	        // B: token rows lane%8, 16-byte column 2kk + (lane/8)%2
   171	        const int brow = lane & 7;
   172	        ldmatrix_x2(b, x + h * (X_BYTES / BOXES) + swz(brow, 2 * kk + ((lane >> 3) & 1)));
   173	        mma(acc, a, b);
   174	      }
   175	    }
   176	    __syncwarp();
   177	    if (lane == 0) mbar_arrive(&empty[s]);
   178	
   179	    const bool tile_end = c == p.chunks - 1 || u == u1 - 1;
   180	    if (!tile_end) continue;
   181	    // flush this CTA's share of tile t: chunks [first_chunk, c]
   182	    const int done = c - first_chunk + 1;
   183	    const bool whole = done == p.chunks;
   184	    const int n0 = t * TILE_N + warp * 16 + g;
   185	#pragma unroll
   186	    for (int i = 0; i < 4; ++i) {
   187	      const int n = n0 + (i >> 1) * 8, m = 2 * tq + (i & 1);
   188	      if (n < p.N && m < p.M) {
   189	        if (whole)
   190	          p.out[static_cast<int64_t>(m) * p.N + n] = __float2bfloat16(acc[i]);
   191	        else
   192	          atomicAdd(&p.ws[static_cast<int64_t>(m) * p.N + n], acc[i]);
   193	      }
   194	      acc[i] = 0.f;
   195	    }
   196	    first_chunk = -1;
   197	    if (whole) continue;
   198	    // count this share in; the CTA completing the tile finishes it
   199	    asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
   200	    if (threadIdx.x == 0) {
   201	      __threadfence();
   202	      last = atomicAdd(&p.counters[t], done) + done == p.chunks;
   203	    }
   204	    asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
   205	    if (last) {
   206	      __threadfence();
   207	      for (int e = threadIdx.x; e < TILE_N * p.M; e += CONSUMERS * 32) {
   208	        const int m = e / TILE_N, n = t * TILE_N + e % TILE_N;
   209	        if (n < p.N) {
   210	          float* src = &p.ws[static_cast<int64_t>(m) * p.N + n];
   211	          p.out[static_cast<int64_t>(m) * p.N + n] = __float2bfloat16(__ldcg(src));
   212	          *src = 0.f;
   213	        }
   214	      }
   215	      if (threadIdx.x == 0) p.counters[t] = 0;
   216	    }
   217	  }
   218	}
   219	
   220	static CUtensorMap make_map(const torch::Tensor& t, int box_rows) {
   221	  // t is [rows, K] bf16 row-major; boxes are 64 elements (128 B) x box_rows
   222	  CUtensorMap map;
   223	  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(t.size(1)),
   224	                              static_cast<cuuint64_t>(t.size(0))};
   225	  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(t.stride(0)) * 2};
   226	  const cuuint32_t box[2] = {HALF_K, static_cast<cuuint32_t>(box_rows)};
   227	  const cuuint32_t unit[2] = {1, 1};
   228	  const CUresult r = cuTensorMapEncodeTiled(
   229	      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, unit,
   230	      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
   231	      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
   232	  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
   233	  return map;
   234	}
   235	
   236	template <int TILE_N, int CHUNK_K>
   237	static void launch(Params p, const torch::Tensor& x, const torch::Tensor& w, int grid_cap) {
   238	  using C = Cfg<TILE_N, CHUNK_K>;
   239	  p.tiles = (p.N + TILE_N - 1) / TILE_N;
   240	  p.chunks = (p.K + CHUNK_K - 1) / CHUNK_K;
   241	  const CUtensorMap wmap = make_map(w, TILE_N);
   242	  const CUtensorMap xmap = make_map(x, MAX_M);
   243	  static bool attr = false;
   244	  if (!attr) {
   245	    cudaFuncSetAttribute(skinny_kernel<TILE_N, CHUNK_K>,
   246	                         cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES);
   247	    attr = true;
   248	  }
   249	  const int64_t units = static_cast<int64_t>(p.tiles) * p.chunks;
   250	  const int grid = static_cast<int>(std::min<int64_t>(grid_cap, units));
   251	  skinny_kernel<TILE_N, CHUNK_K><<<grid, C::THREADS, C::SMEM_BYTES,
   252	                                   at::cuda::getCurrentCUDAStream()>>>(wmap, xmap, p);
   253	}
   254	
   255	// out[M, N] = x[M, K] @ w[N, K]^T; ws fp32 [>= MAX_M * N] and counters int32
   256	// [>= N / 64] must be zero on entry and are left zero. config selects
   257	// (tile rows, K chunk): 0 = 128x128, 1 = 64x128, 2 = 128x256, 3 = 64x256;
   258	// grid caps the CTA count (0 = one per SM).
   259	void gemm(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor ws,
   260	          torch::Tensor counters, int64_t config, int64_t grid) {
   261	  const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
   262	  const int N = static_cast<int>(w.size(0));
   263	  TORCH_CHECK(M >= 1 && M <= MAX_M, "skinny gemm takes 1..8 rows");
   264	  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
   265	                  out.scalar_type() == at::kBFloat16, "bf16 only");
   266	  TORCH_CHECK(w.size(1) == K && x.stride(1) == 1 && w.stride(1) == 1 && out.is_contiguous(),
   267	              "shape / layout");
   268	  TORCH_CHECK(K % 8 == 0 && (x.stride(0) * 2) % 16 == 0 && (w.stride(0) * 2) % 16 == 0,
   269	              "rows must be 16-byte aligned");
   270	  Params p{};
   271	  p.M = M;
   272	  p.N = N;
   273	  p.K = K;
   274	  TORCH_CHECK(ws.numel() >= static_cast<int64_t>(MAX_M) * N &&
   275	                  counters.numel() >= (N + 63) / 64, "workspace too small");
   276	  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
   277	  p.ws = ws.data_ptr<float>();
   278	  p.counters = counters.data_ptr<int>();
   279	  const int cap = grid > 0 ? static_cast<int>(grid) : GRID;
   280	  switch (config) {
   281	    case 0: launch<128, 128>(p, x, w, cap); break;
   282	    case 1: launch<64, 128>(p, x, w, cap); break;
   283	    case 2: launch<128, 256>(p, x, w, cap); break;
   284	    case 3: launch<64, 256>(p, x, w, cap); break;
   285	    default: TORCH_CHECK(false, "unknown config");
   286	  }
   287	  C10_CUDA_KERNEL_LAUNCH_CHECK();
   288	}
   289	
   290	}  // namespace skinny
   291	
   292	PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   293	  m.def("gemm", &skinny::gemm, "bf16 x[M<=8, K] @ w[N, K]^T");
   294	}
```

## mutants/skinny_mut.cu: diff vs production
```diff
--- vllm/model_executor/layers/skinny_gemm/skinny_gemm.cu	2026-09-28 00:55:26.765971000 +0200
+++ agent_space/experiments/2026-10-17-kernel-verification/mutants/skinny_mut.cu	2026-10-17 01:48:45.571169000 +0200
@@ -26,6 +26,9 @@
 #include <algorithm>
 #include <cstdint>
 
+#ifndef MUT
+#define MUT 0
+#endif
 namespace skinny {
 
 constexpr int HALF_K = 64;  // one 128-byte swizzled box width, in bf16
@@ -156,7 +159,7 @@
   for (int u = u0, it = 0; u < u1; ++u, ++it) {
     const int t = u / p.chunks, c = u - t * p.chunks, s = it % STAGES;
     if (first_chunk < 0) first_chunk = c;
-    mbar_wait(&full[s], (it / STAGES) & 1);
+    if (!(MUT == 1 && it == 0)) mbar_wait(&full[s], (it / STAGES) & 1); if (MUT == 2 && lane == 0) mbar_arrive(&empty[s]);
     const uint32_t w = smem_u32(ring + s * STAGE_BYTES);
     const uint32_t x = w + W_BYTES;
 #pragma unroll
@@ -174,7 +177,7 @@
       }
     }
     __syncwarp();
-    if (lane == 0) mbar_arrive(&empty[s]);
+    if (MUT != 2 && lane == 0) mbar_arrive(&empty[s]);
 
     const bool tile_end = c == p.chunks - 1 || u == u1 - 1;
     if (!tile_end) continue;
@@ -201,7 +204,7 @@
       __threadfence();
       last = atomicAdd(&p.counters[t], done) + done == p.chunks;
     }
-    asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
+    if (MUT != 3) asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
     if (last) {
       __threadfence();
       for (int e = threadIdx.x; e < TILE_N * p.M; e += CONSUMERS * 32) {
```

## The bfi site in the PTX before/after prep
```
before: bfi.b32 %r298, %r569, %r297, 4, 28;   (%r297 = %r1 & 15)
	{ .reg .b32 %bfa, %bfb;
	shl.b32 %bfa, %r569, 4;
	and.b32 %bfa, %bfa, 4294967280;
	and.b32 %bfb, %r297, 15;
	or.b32 %r298, %bfa, %bfb; }
```

## sweep.sh
```bash
#!/usr/bin/env bash
# compute-sanitizer over our Hopper kernels' correctness suites, login GPU.
# Usage: sweep.sh [tools...]  (default: racecheck synccheck memcheck initcheck)
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh >/dev/null 2>&1
export PYTHONPATH=$PWD
E=agent_space/experiments/2026-10-17-kernel-verification
L=$E/logs/sweep; mkdir -p $L
T=tests/kernels/moe
declare -A CMD FILT
CMD[skinny]="$E/skinny_check.py";                           FILT[skinny]=skinny::
CMD[sliced]="-m pytest -q -x $T/test_tiered_decode_sliced.py -k 1-False||8-True||47-True||64-True"
FILT[sliced]=tiered_decode::
CMD[decode]="-m pytest -q -x $T/test_tiered_decode_moe.py -k 8-int4||32-mxfp4||padding||replica"
FILT[decode]=tiered_decode::
CMD[prefill]="-m pytest -q -x $T/test_tiered_prefill_moe.py -k sliced_tiers_match_fp32||two_tiers_match_fp32||heavy||padding||grace||tiny||schedules"
FILT[prefill]=tiered_prefill::
for tool in ${@:-racecheck synccheck memcheck initcheck}; do
  for s in skinny sliced decode prefill; do
    extra="--kernel-name kns=${FILT[$s]}"
    [[ $tool == memcheck ]] && extra="$extra --leak-check no"
    # a kernel filter hides other kernels' writes from initcheck, so every
    # torch-produced buffer would read as uninitialized
    [[ $tool == initcheck ]] && extra=""
    read -r -a argv <<< "${CMD[$s]}"
    for i in "${!argv[@]}"; do argv[$i]=${argv[$i]//||/ or }; done
    start=$(date +%s)
    compute-sanitizer --tool $tool $extra --print-limit 20 \
      .venv/bin/python "${argv[@]}" > $L/$s.$tool.log 2>&1
    rc=$?
    echo "$(date +%T) $s $tool rc=$rc $(( $(date +%s) - start ))s $(grep -h -E 'SUMMARY' $L/$s.$tool.log | tail -1) $(grep -h -E '[0-9]+ (passed|failed)' $L/$s.$tool.log | tail -1)" >> $L/summary.txt
  done
done
echo DONE >> $L/summary.txt
```

## skinny_check.py
```python
"""skinny_gemm correctness driver for compute-sanitizer runs.

Covers every config at shapes where a CTA owns whole tiles, where split-K
spreads one tile over several CTAs (the atomics + last-CTA flush path), and a
ragged N/K edge. Asserts against torch and that the zero-on-exit workspace
contract holds.
"""

import sys

import torch

from vllm.model_executor.layers.skinny_gemm import _workspace, skinny_gemm

torch.manual_seed(0)
dev = torch.device("cuda:0")
cases = [
    # (M, N, K, config, grid)
    (8, 128, 512, 1, 2),  # whole tiles per CTA
    (8, 128, 512, 1, 3),  # uneven split-K
    (5, 256, 1024, 0, 7),
    (1, 192, 768, 2, 5),
    (8, 200, 520, 3, 4),  # ragged N and K
    (8, 6144, 2048, 0, 0),  # one CTA per SM
]
bad = 0
for m, n, k, cfg, grid in cases:
    x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
    for _ in range(2):
        got = skinny_gemm(x, w, cfg, grid).float()
    want = x.float() @ w.float().t()
    err = ((got - want).abs().amax() / want.abs().amax()).item()
    ws, counters = _workspace(dev, n)
    dirty = int(ws.count_nonzero()) + int(counters.count_nonzero())
    ok = err < 1e-2 and dirty == 0
    bad += not ok
    print(f"M={m} N={n} K={k} cfg={cfg} grid={grid}: err={err:.2e} dirty={dirty}",
          "ok" if ok else "FAIL")
sys.exit(1 if bad else 0)
```
