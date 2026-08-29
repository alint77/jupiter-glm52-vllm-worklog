# GLM-5.3 W4A16 (group 32) against NVFP4

Benchmarks `JANGQ-AI/GLM-5.3-W4A16` on the tiered MoE path and compares it to
the incoai NVFP4 checkpoint that has been the production target.

## The checkpoint

compressed-tensors `pack-quantized`, int4 symmetric, **group_size 32**,
`actorder: None`, targeting `re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$`
with `lm_head`, embeddings, norms, `self_attn`, `shared_experts`, `mlp.gate`,
the dense MLPs and `eh_proj` all ignored. 282 shards, 420 GB, 176,321 tensors.

Two things had to change before it would load.

### group 32 was rejected by three hardcoded guards

`tiered_moe_manifest.py`, `tiered_moe_conversion.py` (twice) and
`compressed_tensors_moe_wna16_marlin.py` all required `group_size == 128`.
Nothing depended on that value:

- Marlin supports it -- `quant_utils.SUPPORTED_GROUP_SIZES` is `[-1, 32, 64, 128]`.
- `_w2_scale_sharding` branches only on `actorder`, never on the group size.
- `runtime_expert_bytes` is derived from the stored tensor sizes, so a smaller
  group simply costs more resident scale bytes and is accounted automatically.

All three now accept Marlin's set. 43/43 manifest tests pass and the existing
checkpoints still resolve unchanged (Int4-Int8Mix g128, NVFP4 g16).

### the published index declares the wrong total_size

`total_size` is 450,925,945,160 against 450,904,205,312 of actual tensor
headers, a 21,739,848 byte overstatement, which trips the truncation guard in
`build_glm_w4a16_manifest`.

The checkpoint is not truncated: 176,321 tensors in the index, 176,321 on disk,
none missing, none extra, and every shard byte-exact against the hub manifest.
So this is publisher bookkeeping. The local metadata is corrected to the summed
value rather than weakening a guard that is doing its job; the published
original is kept at `index-original.json`.

## Footprint against NVFP4

| | W4A16 g32 | NVFP4 |
| --- | --- | --- |
| runtime_expert_bytes | 21,233,672 | 21,233,680 |
| routed_expert_bytes | 407.7 GB | 407.7 GB |
| non_routed_bytes | **43.2 GB** | **57.1 GB** |

Per-expert cost is within 8 bytes: int4 with bf16 scales at group 32 costs the
same as fp4 with fp8 scales at group 16. The difference is **13.9 GB less
non-routed weight, about 3.5 GB per rank**, because this checkpoint quantizes
the MTP block (768 packed expert tensors at layer 78) where NVFP4 leaves it
BF16 at roughly 4.5 GiB per rank.

## Method

`arm-quant-ab.sh` is the real-code decode suite -- 512-token code prompts,
1024-token generations -- which is the only suite in this tree that measured the
placement ranking honestly: the 16K suite contaminates TPOT with chunked-prefill
stalls and the synthetic `random` dataset cannot test anything content-derived.

Both arms use the re-derived GLM-5.3 ranking at 2400 hot slots per rank, so the
placement, concurrency, speculator and prompts are fixed and the checkpoint is
the only variable. `restamp_profile.py` re-fingerprints the profile onto the
W4A16 checkpoint without trimming slots or relabelling the ranking, which is
what `port_profile.py` would have done.

Three paired runs per side: between-run spread on this configuration is about
2.7%, wider than any within-run confidence interval, so a single pair cannot
size a small effect.

**At a fixed 2400 slots the MTP saving shows up as spare HBM, not throughput.**
This run therefore isolates format and kernel speed. If W4A16 holds up, the
follow-up is to spend those 3.5 GB per rank on more hot slots.

## Four things blocked the load

Each was found by a job failing on the cluster, so they are listed in the order
they surfaced. Only the last is a genuine bug; the rest were guards stricter
than the kernel.

1. **`group_size == 128` in four places** (manifest, both conversion gates,
   Marlin construction). Marlin supports `[-1, 32, 64, 128]`,
   `_w2_scale_sharding` branches only on `actorder`, and `runtime_expert_bytes`
   derives from stored sizes. Widened to Marlin's set.
2. **`total_size` truncation guard.** The published index overstates it by
   21,739,848 bytes. The checkpoint is complete -- 176,321 tensors in the index
   and on disk, no missing or extra, every shard byte-exact against the hub --
   so the local metadata was corrected rather than the guard weakened.
   `index-original.json` keeps the published version.
3. **Pinned per-component `(shape, dtype)` tables.** `weight_scale` exists in
   both the compressed-tensors and NVFP4 layouts, so shape is the
   discriminator, and group 32 moves `down_proj.weight_scale` from (6144, 16)
   to (6144, 64). Adding one table per group was not enough on its own: the
   group tables share identical `weight_packed` and `weight_shape` entries, so
   those components matched three tables at once and tripped the ambiguity
   guard (three tests caught this). The stager now narrows candidates as
   components arrive and requires a unique survivor at completion; every expert
   carries three `weight_scale` tensors, so uniqueness is always reached.
4. **`allocate_layer_expert_storage(group_size: int = 128)`** -- a real latent
   bug. AutoRound and ModelOpt pass the group size explicitly from their quant
   config; the compressed-tensors path alone took the default. Invisible while
   every such checkpoint was group 128, but it sized the final Marlin storage
   for 128 against group-32 weights. The default is removed so every call site
   must be explicit.

Storage layouts now agree with the manifests at every group size: g16
21,233,680, g32 21,233,672, g64 20,054,024, g128 19,464,200.

## Results

NVFP4 baseline, three runs, c4/DCP4/MTP3 on the real-code decode suite:

| run | aggregate | per-request |
| --- | --- | --- |
| nvfp4-r1 | 189.0 | 47.2 |
| nvfp4-r2 | 196.1 | 49.0 |
| nvfp4-r3 | 197.6 | 49.4 |

**194.2 +/- 4.7 tok/s aggregate.** This reproduces the 197.3 +/- 6.0 measured
for the same checkpoint in the routing-capture experiment, so the baseline is
stable across sessions and nodes.

W4A16 pending: `1534564`-`1534569` failed on issue 3, `1534577`-`1534579` on
issue 4, `1534697`-`1534699` are the third attempt.

## Prefix caching degrades the first pass (pre-existing, both checkpoints)

GSM8K five-shot, 256 questions, temperature 0, seed 42 -- so every repeat sends
**identical questions greedily** and must produce identical answers.

| arm | prefix caching | r1 | r2 | r3 |
| --- | --- | --- | --- | --- |
| W4A16 g32 | on | **0.586** | 0.914 | 0.910 |
| W4A16 g32 | off | 0.914 | 0.926 | 0.914 |
| NVFP4 g16 | on | **0.563** | 0.906 | 0.910 |

Reproduced independently at 0.594 in an earlier run, so it is deterministic.

Established:

- **Prefix caching is necessary.** With `--no-enable-prefix-caching` the first
  pass is clean, on the same server, after the same warmup.
- **Not checkpoint-specific.** NVFP4 -- the previous production target -- fails
  identically, so this is not the group-32 work in Phase 46.
- **Only the first pass.** r2 and r3 are healthy with caching still enabled.
- **The 8-question warmup does not prevent it**, and may be causing it: the
  warmup shares the same five-shot prefix, so a wrong first forward would be
  cached and inherited by all 256 of r1's questions. r2 recovers because 256
  unique suffixes evict those blocks and the prefix is recomputed warm.
- **Degraded runs ramble.** 26.7k-37.3k output tokens against 23.5k-24.3k when
  healthy -- off-distribution continuation, not random wrong answers.

Two hypotheses remain, with different fixes:

1. **MTP draft metadata against cached tokens.** The fork patches exactly this
   code (`llm_base_proposer.py` invalidates `_num_computed_tokens_cpu` at line
   678 and increments it at 815), and prefix caching is what makes that value
   jump. `acc-nospec-pc` tests it by removing the speculator.
2. **Cold-tier first touch poisoning the cache.** Fix would be warming before
   anything is cacheable, and disabling caching would be the *wrong* remedy.

`repro_prefix_cache.py` separates them: it sends a long shared prefix cold, then
re-sends identical prompts warm, and diffs the greedy output. Any difference is
a cache-correctness bug.

**Production impact is not yet established.** This eval is 256 independent
requests sharing one static prefix; a Claude Code session is a single
conversation extending its own prefix. Those exercise the cache differently.
What is certain is that prod runs with prefix caching enabled and starts a
fresh server per session, so it lands in the affected window.

## Root cause narrowed, not closed

| speculator | prefix cache | warmup | first-pass GSM8K |
| --- | --- | --- | --- |
| MTP3 | on | yes | 0.586 / 0.563 (NVFP4) |
| MTP3 | on | **no** | 0.582 |
| MTP3 | off | yes | 0.914 |
| none | on | yes | 0.906 / 0.914 / 0.906 |

Both MTP and prefix caching are required. The 8-question warmup is irrelevant
(0.582 without it), which kills the cache-poisoning hypothesis. The tiered path
is equally cold in the no-speculator arm, so cold-tier first touch is out too.

### The losslessness test is blocked by nondeterminism

MTP is lossless by construction at temperature 0, so an accuracy drop implies
the verifier accepts tokens the target would not emit. Comparing completions
between an MTP server and a no-speculator server gave 63/64 differing -- but
**the control gives 60/64 differing between two passes on the same MTP
server**. This build is not reproducible run to run at temperature 0, so text
diffing cannot see the effect.

That also retracts `repro_prefix_cache.py`'s finding: what it labelled "prefix
cache changes greedy output" was this same nondeterminism, and its verdict
string fires on the arm where caching is disabled.

Accuracy over 256 questions is a statistical measure and survives per-token
jitter -- a 33pp gap is not formatting noise -- so the effect is real even
though the token-level test cannot resolve it.

Closing it needs logit-level instrumentation: capture the target's argmax at
each accepted position and assert it equals the accepted token. That is an
engine patch, not another benchmark.

### Separately

Greedy decoding is not reproducible across identical requests (60/64 differ on
one server, temperature 0). That is a defect in its own right and independent
of everything above.
