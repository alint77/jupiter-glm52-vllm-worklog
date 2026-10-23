# MLA qkv glue in one kernel, served (2026-10-18)

The user: "lets pursue the glue2 work" (task #102). glue2 (`mla_glue.cu`,
vllm `vllm/model_executor/layers/mla_glue/`, `VLLM_MLA_QKV_GLUE=1`) does MLA's
work between `fused_qkv_a` and `q_b` in one CTA-per-token kernel: the q_a
(2048) / kv_a (512) RMSNorms (fp32 sums, one bf16 rounding, as inductor), the
k_pe interleaved RoPE, and the layer's fp8_ds_mla KV writes (main cache, and
the skip-KV stage's reserved rows on index-share layers). q RoPE stays where
it was. PDL: `launch_dependents` + `wait` at entry (decode_gemm loads weights
before its `griddepcontrol.wait` and activations after; cuBLAS above M=16 does
not take part).

`tests/kernels/test_mla_qkv_glue.py` (9 pass): k_pe bitwise vs an fp32
reference, qn / kvn <= 1 ulp on < 1e-4 of elements; cache bytes equal
`concat_and_cache_mla` of the kernel's own outputs for both caches (padded
slot, rows >= 128 = -1, fewer slots than tokens, sentinels); no-cache path;
`writes_kv` true only inside a capture with attention metadata.

## Three designs

1. **Splitting op** (`glue-v1-splitting.diff`, run mg*, `logs/failed-b/`):
   `mla_qkv_glue` as a splitting op so piecewise steps resolve caches at run
   time. Piecewise capture crashed, "capturing stream has unjoined work": the
   new split put a side-stream fork and its join (cold prefetch / indexer) in
   different pieces. Startup also grew to ~40 min.
2. **KV only in FULL graphs, keyed on `cudagraph_runtime_mode == FULL`**
   (run mg2, `pf7`). Started, captured and served cleanly, but the glue never
   wrote the KV: the V2 runner captures FULL graphs by running a
   `cudagraph_runtime_mode=NONE` forward inside `torch.cuda.graph`
   (`vllm/v1/worker/gpu/cudagraph_utils.py:345`). Traces: `qkv_glue_kernel`
   once per layer, the triton norm / k-RoPE kernels gone, but
   `concat_and_cache_ds_mla_kernel` at prod's count (1.70 per attention call
   both arms). So mg2 b = **norms + k RoPE only**.
3. **`mla_glue.writes_kv()`** = stream capturing and attention metadata
   present (FULL capture only: piecewise captures have none, eager steps and
   piecewise replays are not capturing). The op and the kv-cache-update skip
   both use it, so a forward never writes twice or not at all. Run mg3
   (`pf8`, `launch_ab3.sh`).

## Served A/B

ABBA on 8 holds (2396112-19), c=4 launcher config, a = prod (pf6, touch-order
profile), fresh compile caches per server; per hold both arms back to back,
order alternating. `ab_compare.py <run>`: verify-step medians (graph replays
>= 500 kernels, device 0), b - a on the same hold; "rest" = span - MoE
(`layer_kernel`), the part the glue can move (MoE swings with the traffic's
cold experts).

Run 2, b = norms + k RoPE (`ab_compare_mg2.txt`; ms):

| arm | M | rest b-a per hold | mean | span b-a mean |
|---|--:|---|--:|--:|
| dec1 ~5K | 4 | -0.25, -0.43 | -0.34 | -0.32 |
| dec1 ~50K | 4 | -0.23, -0.44 | -0.33 | -1.16 |
| dec4 ~5K | 16 | -0.19, -0.13 | -0.16 | -0.03 |
| dec4 ~50K | 16 | -0.30, -0.09 | -0.19 | -0.39 |
| c=4 swarm | 16 | -0.26, -0.41 | -0.34 | +0.72 |

10 of 10 pairs negative outside MoE; prod's M=4 step is ~14.3 ms. Two of four
swarm holds caught a prefill phase in one arm's second window (9 verify
steps) and drop out.

Run 3 (`launch_ab3.sh`, glue writes the KV): the first c trace has no
`concat_and_cache_ds_mla_kernel` at all (`qkv_glue_kernel` 3.8 -> 4.5 us),
but stopped after one pair: **the glue arms decode worse.**

## Decode quality regression (fixed 2026-10-23)

MTP acceptance on the dec1 arms: prod 3.43 / 3.14, run-2 b 2.01 / 2.03, run-3
c 1.64. pf6 and pf7 differ only by glue.diff. Op checks on the login GPU
(correctness only): kernel vs vLLM's own `RMSNorm.forward_native` and
`RotaryEmbedding.forward_cuda` (k_pe bitwise, norms 1 ulp); q RoPE with
`key=None` on the strided slice = with k. PDL overlap is real (`q_b`'s
decode_gemm starts before glue ends in every step; decode_gemm loads weights
before `griddepcontrol.wait`, activations after; its own `launch_dependents`
precedes its y stores, which the dependent's wait covers).

Bisection (`launch_probe.sh`, `probe.py`, `probe_arm.sh`; one server per
hold at the c=4 config; pf9 = pf8 + `debug-knobs-pf9.diff`; probe = prompt
logprobs of a 3K prompt, greedy 4 (mgp) / 8 (mgq) x 5K chats of 300 tokens,
sampled 4 x 5K; `probe_summary.txt`):

| variant | servers | greedy lp | sampled lp |
|---|--:|---|---|
| prod (pf6) | 4 | -0.123 .. -0.128 | -0.236 .. -0.290 |
| glue (pf9) | 4 | **-0.171 .. -0.210** | **-0.546 .. -0.600** |
| krope: glue, downstream k_pe from rotary_emb | 3 | -0.116 .. -0.137 | -0.237 .. -0.318 |
| kvorig: + kv_c_normed from the module | 1 | -0.124 | -0.246 |
| qorig: q_c from the module, glue's kv outputs | 1 | -0.130 | -0.308 |
| nopdl / nodraft / both | 1 each | -0.177 / -0.232 / -0.172 | -0.312 / -0.387 / -0.277 |

- Prefill is clean: prompt-logprob mean |d| glue vs prod 0.061-0.078 = prod
  vs prod 0.065 (serving is not bitwise deterministic; greedy outputs of two
  prod servers diverge within 2 tokens too).
- Decode is not: every full-glue server is less confident, by far the most
  under sampling.
- Not the math: krope changes only the k_pe *tensor* glue hands on, and
  nothing reads it in a decode step (MQA attention reads q and the cache; the
  cache update is skipped; glue writes the cache from its in-kernel values in
  both). qorig changes only q_c. Each alone restores prod. Neither PDL nor
  the MTP layer accounts for it (nopdl / nodraft stay bad).
- So likely ordering / lifetime: something reads glue's outputs (or what it
  writes) while it is still changing, and any extra kernel in between narrows
  the window. Leads: the custom op allocates qn / kvn / kpe itself (inside the
  compiled pieces and the FULL capture), so their reuse is the caching
  allocator's, not inductor's; the side streams in the layer (cold-prefetch
  fork/join, skip-KV stager, L2 prefetch). Next probes: `--enforce-eager`
  glue vs prod; glue with outputs passed in from the caller (inductor-owned);
  compute-sanitizer racecheck on one replayed step.

The perf numbers above (run 2) measure a path that decodes worse; they are
not a ship case.

### Root cause: decode_gemm reads x before its PDL wait (2026-10-23)

The source loads activations after `griddepcontrol.wait`, but they use
`__ldg` (`ld.global.nc`), and nvcc hoists non-coherent loads above the wait:
in the served `.so` every PDL instantiation issues 2-32
`LDG.E.128.CONSTANT` x loads before `ACQBULK`. Glue calls
`launch_dependents` at its entry, so `q_b`'s decode_gemm starts ~1.4 µs
after glue and reads part of q_c before glue writes it. This fits all of the
bisection: decode only (prefill uses F.linear); op tests pass (no PDL
neighbour); qorig and krope put a kernel between glue and `q_b`; nopdl kept
glue's `launch_dependents` and `q_b`'s PDL launch. Found by reading the code
without a GPU (subagent + Astra, `astra/race-review.md`); Astra's top
candidate was storage reuse under PDL.

Fix (`ldx`, worktrees pf12 = pf6 + fix, pf13 = pf9 + fix): x through a
volatile `ld.global.v4.u32`, which stays below the wait. Weights keep
`__ldcs` before it.

- SASS: in all 24 PDL instantiations, no load other than the weights'
  `LDG.E.EF.128` comes before `ACQBULK`.
- `race_test.py` + `producer.cu` (login GPU, correctness only): the producer
  triggers dependents at entry and writes x after 20K cycles; a PDL
  decode_gemm on x is then compared bit for bit with the serialized run, at
  q_b / qkv_a, M = 4/8/16/32, all 6 configs, eager and graph. Old: 9,582 of
  9,600 runs differ. Fixed: 0.
- Other PDL consumers, same SASS check: sliced decode `layer_kernel` /
  `route_prep` / `finalize` and the glue kernel load nothing before their
  wait. `tiered_prefill` `gemm_xres_kernel` has 1, by design: only its tail
  CTA waits.
- Prod has the same hoist. Its fused_qkv_a decode_gemm follows the
  allreduce-fusion kernel with PDL; whether prod reads early depends on
  where that kernel triggers. The served probes (`launch_dgfix.sh`) test it.

## Astra (`astra/integration-review.md`, on design 1)

- P1 PDL: marked resolved because decode_gemm's source reads activations
  after its wait (`decode_gemm.cu:69-71`). Wrong: the SASS hoists them
  above it (root cause above). cuBLAS consumers do not take part.
- P1 splitting registration: moot after design 1.
- P2 eligibility: `supported()` requires `FlashMLASparseImpl`, the exact
  RotaryEmbedding / RMSNorm types and dims, fp8_ds_mla.
- Open: position contract (only a negative wrap; no bounds check). Served
  equivalence: failed in decode (above).

The vLLM code (`glue.diff`: mla_glue/, mla.py, mla_attention.py, envs.py,
the test) stays uncommitted in the main tree, off by default.

### Scope gate (2026-10-23)

The user: glue only where it is tested extensively. `supported()` now also
requires the server's largest decode step, max_num_seqs x (1 + speculative
tokens), to be <= 16 (`MAX_DECODE_TOKENS`): the c=4 MTP3 config (4 x 4) and
smaller. c=8 (32) and c=16 (64) keep the module path. The gate is per server
because a token-count branch in the forward would be traced once by
torch.compile, not evaluated per step. Main tree only (uncommitted), along
with the decode_gemm fix; the probe worktrees pf12 / pf13 are unchanged.

Kernel time, old / new / clob (`bench_dgfix.py`, Booster hold 2473004, GPU 0
NUMA-bound, L2 flushed, 2 runs x 5 alternations x median of 50; the five
served shapes at M = 4..32 with decode_gemm.config's pick, PDL on and off):
every new - old and clob - old is within -0.31 .. +0.31 us on 8-29 us
kernels, no sign either way (`bench_dgfix-2473004.jsonl`). The clobber
(Astra's hardening) costs nothing, so the main tree takes `clob.cu`; the
probe worktrees run `new.cu` (same SASS ordering, both 0 in race_test).

### Served check of the fix (`launch_dgfix.sh`, `probe_dgfix.txt`)

probe.py at the c=4 config, 4 holds (2471698, 2472833-35), 2 servers each:
p = prod (pf6), q = prod + fix (pf12), f = glue + fix (pf13).

| arm | servers | acc greedy | acc sampled | greedy lp | sampled lp |
|---|--:|---|---|---|---|
| prod | 3 | 3.03-3.13 | 2.83-2.94 | -0.121 .. -0.131 | -0.256 .. -0.341 |
| prod + fix | 2 | 3.07-3.10 | 2.89-2.99 | -0.131 .. -0.137 | -0.270 .. -0.280 |
| glue + fix | 3 | 3.06-3.12 | 2.87-2.97 | -0.125 .. -0.129 | -0.259 .. -0.291 |
| glue, unfixed (above) | 4 | 3.00-3.17 | 2.48-2.66 | -0.171 .. -0.210 | -0.546 .. -0.600 |

The fix restores glue to prod. Prompt logprobs glue + fix vs prod on one hold:
mean |d| 0.066 (prod vs prod 0.065). Prod + fix stays within prod's spread, so
prod's own hoisted loads (fused_qkv_a after the allreduce-fusion kernel) did
not show in output.

Committed (vllm, local, not pushed): bec3187c33 decode_gemm fix with
`test_pdl_reads_x_after_wait` in tests/kernels/test_decode_gemm.py (fails 4/4
on the old kernel, passes on the fix); d5e946cc43 glue, off by default, with the
<= 16 decode-token server gate. Speed: run 4 (`launch_ab4.sh`, worktrees/glue-ab
= d5e946cc43, glue off vs on, dec1 / dec4 / tlnsys, ABBA on 6 holds).

### Run 4: glue does not start on current prod (2026-10-23)

All three glue-on arms (`mg4g-*`, worktrees/glue-ab = d5e946cc43) died in
FULL graph capture: `AssertionError: anchor hot cache map without its KV
write` (anchor_hot_cache.py:158), then "capturing stream has unjoined work".
The anchor hot cache (e9d65055e3..e51272479d, on by default in serve.sh) came
after the probe base (pf6 = 65d9b56fb1): on anchor layers
`do_kv_cache_update` also writes the step's tokens into the hot cache's
reserved rows and arms `expect_map`. The glue writes the main cache and the
skip-KV rows, then skips `do_kv_cache_update`, so the anchor's map finds no
write. So the probes validated glue on a base without the anchor hot cache,
and the committed glue is broken on current prod (it fails loudly, at
capture). The glue-off arms ran (nsys reps not exported).

Fix (option 1, the user's pick): vllm 569c42871c. `writes_kv(impl)` is false on
hot-cache anchors (`impl._hot_anchor`), so there the glue only normalizes and
rotates and `do_kv_cache_update` writes main + hot cache as in prod; the 57
index-share layers keep the fused write. Glue tests 9 pass (the writes_kv test
covers the anchor case). Requalification on current prod from
worktrees/glue-ab2: quality `launch_probe5.sh` (pq5-*, 3 holds, a = glue off,
g = on) and speed `RUN=mg5 TREE=glue-ab2 launch_ab4.sh` (6 holds).

### Requalified on current prod (2026-10-23, worktrees/glue-ab2 = 569c42871c)

Quality (`launch_probe5.sh`, `probe_pq5.txt`; a = glue off, g = on, 3 holds):

| arm | servers | acc greedy | acc sampled | greedy lp | sampled lp |
|---|--:|---|---|---|---|
| glue off | 3 | 2.98-3.06 | 2.81-2.88 | -0.133 .. -0.150 | -0.257 .. -0.318 |
| glue on | 3 | 3.02-3.09 | 2.76-2.88 | -0.135 .. -0.141 | -0.276 .. -0.344 |

Prompt logprobs g vs a per hold: mean |d| 0.065 / 0.074 / 0.069 (prod vs prod
0.065). All glue servers captured and served.

Kernels per M=16 verify step (device 0): off 156 `concat_and_cache_ds_mla`;
on 78 `qkv_glue_kernel` + 42 `concat_and_cache_ds_mla` (21 anchors x main +
hot cache).

Speed (`RUN=mg5 TREE=glue-ab2 launch_ab4.sh`, `ab_compare_mg5.txt`; g - a on
the same hold, medians of verify steps; rest = span - MoE):

| kind | M | prod step | rest g-a per hold | mean | span g-a mean |
|---|--:|--:|---|--:|--:|
| dec1 ~5K | 4 | 14.1 ms | -0.52, -0.64 | -0.58 | -0.50 |
| dec1 ~50K | 4 | 14.4-15.5 | -0.56, -0.64 | -0.60 | -0.54 |
| dec4 ~5K | 16 | 22.0-22.3 | -0.30, -0.33 | -0.31 | -0.34 |
| dec4 ~50K | 16 | 22.3-22.7 | -0.32, -0.40 | -0.36 | -0.20 |
| c=4 swarm | 16 | 22.1-23.1 | -0.32, -0.01 | -0.16 | -0.54 |

11 of 12 pairs faster outside MoE (one -0.01). ~4% of the M=4 step, ~1.5% at
M=16.
