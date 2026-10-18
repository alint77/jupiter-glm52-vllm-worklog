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

## Decode quality regression (open)

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

## Astra (`astra/integration-review.md`, on design 1)

- P1 PDL: resolved, decode_gemm reads activations after its wait
  (`decode_gemm.cu:69-71`); cuBLAS consumers do not take part.
- P1 splitting registration: moot after design 1.
- P2 eligibility: `supported()` requires `FlashMLASparseImpl`, the exact
  RotaryEmbedding / RMSNorm types and dims, fp8_ds_mla.
- Open: position contract (only a negative wrap; no bounds check). Served
  equivalence: failed in decode (above).

The vLLM code (`glue.diff`: mla_glue/, mla.py, mla_attention.py, envs.py,
the test) stays uncommitted in the main tree, off by default.
