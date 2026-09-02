# Phase 51: upstream sglang reaches 5.71 on DFlash2 where this fork reaches 3.995

**This overturns Phase 48's verdict.** That phase concluded "DFlash2 is now working
as well as this implementation can make it work... the card's 5.94 is not
reachable by fixing fork defects, because there are none left that we can find."
Measured on the same hardware, the same checkpoint pair and the same protocol,
stock upstream sglang gets within 4% of the card. **The fork has a DFlash2
defect worth ~43% acceptance, and there is now a working reference to diff
against.**

## Result

GSM8K, card protocol (chat template, natural EOS, T=1.0 / top-p 0.95, 4096 max
new tokens, 64 samples at concurrency 1), incoai NVFP4 target + incoai DFlash2
drafter, block size 8 = 7 draft tokens.

| stack | AL mean | median | accept rate | decode tok/s | % of card |
| --- | ---: | ---: | ---: | ---: | ---: |
| card (inco, SGLang / 4x GB300 / FA4) | 5.94 | - | - | - | 100% |
| **sglang upstream, draft attn `triton`** | **5.7236** | 5.7278 | 0.6741 | 73.8 | **96.4%** |
| **sglang upstream, draft attn `flashinfer`** | **5.7063** | 5.7048 | 0.6716 | 72.7 | **96.1%** |
| **this fork** (Phase 48 `fixedgreedy-gsm8k`) | **3.9951** | - | - | - | **67.3%** |

Two independent draft-attention backends agree to **0.30%**, so 5.71 is a
property of the implementation rather than of a kernel choice, and run-to-run
spread on this measurement is well under 1%.

## The control that makes it interpretable

The same harness, same server, same protocol, on the target's own MTP head:

| | fork | sglang | delta |
| --- | ---: | ---: | ---: |
| MTP7 acceptance length | 4.9348 | 4.9489 | **0.29%** |
| MTP3 accept rate | 0.798 (production, recorded) | 0.8050 | 0.9% |

**MTP agrees across the two engines to 0.29%.** So the DFlash2 divergence is
not the hardware (GH200 vs GB300), not the checkpoint, not the protocol, not
the harness, and not a general fork-vs-sglang difference. It is specific to
DFlash2.

This retires, by measurement, every alternative explanation the earlier phases
proposed for the 32% gap:

| earlier hypothesis | status now |
| --- | --- |
| GB300 / FA4 / SGLang stack advantage (48) | **refuted** - sglang on GH200 with `triton` draft attention gets 5.71 |
| The drafter is simply weak on math against this target (49e) | **refuted** - same weights, same target, 5.71 |
| The untrained mask-token embedding row (49d) | **refuted** - sglang reads the same row from the same released weights |
| Checkpoint revision (49e) | already refuted; both stacks run `bae18bbff1` |
| Target quantisation (50) | already refuted; both stacks run incoai NVFP4 at commit `54e5252` |

Phase 49f localised the fork's entire deficit to **position-0 candidate
recall** (64.1% on GSM8K vs 98.4% on HumanEval, with position-0 acceptance =
recall x the target's mode-agreement rate to first order). sglang's drafter,
given the same weights and the same target, evidently places the true token in
its top-16 far more often. **The fault is in what our draft is fed, or in how
its candidates are computed** - and `srt/models/dflash.py` plus
`srt/speculative/dflash_*.py` are the reference for the diff.

## Throughput: DFlash2 wins, and not for the reason the vendor's blog implies

| arm | decode tok/s (median) | AL | step ms | vs MTP3 | vs MTP7 | wall for 64 reqs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP3 | 50.9 | 3.418 | 67.1 | 1.00x | 0.80x | 541 s |
| MTP7 | 63.8 | 4.949 | 77.5 | 1.25x | 1.00x | 481 s |
| DFlash2 (flashinfer) | 72.7 | 5.706 | 78.4 | 1.43x | 1.14x | 215 s |
| DFlash2 (triton) | 73.8 | 5.724 | 77.5 | **1.45x** | **1.16x** | 228 s |

Step time from Phase 42's own cost model, `tok/s = acceptance / step_time`.

**DFlash2 and MTP7 have identical step times (77.5 ms).** DFlash2's entire win
is acceptance; its draft is *not* cheaper on this stack. That contradicts the
premise of the parallel drafter and contradicts inco's published figures, from
which a ~8% cycle advantage can be derived (GSM8K speedup ratio 3.23/2.58 =
1.252 against AL ratio 5.94/5.12 = 1.160). Here the same ratio is 1.000.

Plausible cause, **untested**: 2-node TP8 puts every collective on InfiniBand
rather than NVLink, and a wider draft block needs more of them, cancelling the
compute advantage. On their single-node GB300 it survives. Testing this needs a
single-node sglang run, which needs tiering sglang does not have.

This is also the first direct evidence on the question the worklog has carried
as open since Phase 42 - *"matched MTP3-vs-DFlash2 end-to-end throughput on a
5.3 target, never been run"*. On this stack DFlash2 wins by 45% over MTP3. But
note what that depends on: at the **fork's** acceptance of 3.995, the same step
time would put DFlash2 *below* MTP3. The throughput win exists only because
sglang gets 5.71.

## Absolute throughput is not comparable to the fork

Every number here is TP8 across two nodes: NVLink only within GPU pairs,
everything else over IB. The fork's single-node c1 decode is 101-108 tok/s, so
roughly 2x these figures - which is what the topology predicts and is not
evidence about sglang. The card's 366.6 tok/s is a third stack on a newer GPU
generation. **Only the within-table comparisons are valid, and acceptance is
the transport-invariant quantity.**

## Method

Fresh tree at `/e/fscratch/profound/naeimitabiei1/sglang-fresh-20260902`,
deliberately not either local fork:

- upstream `sglang 0.5.6.post3.dev9882+gdb017e349` (main), built from git
- `sgl_kernel` 0.4.6.post1 and `flashinfer` 0.6.18 as aarch64 wheels (no source
  build, so the DFlash2 selector does **not** fall back to `torch.topk`)
- its own venv with exactly one sglang tree; `sglang_tree_check` runs per job
- harness `bench_al.py` reads sglang's own per-request
  `meta_info["spec_accept_length"] = completion_tokens / spec_verify_ct`
  (`tokenizer_manager.py:2814`) - byte for byte the card's definition - rather
  than reconstructing it from counters

Server, 2 nodes x 4 GH200, TP8:

```
--tp-size 8 --nnodes 2 --node-rank $rank --dist-init-addr $head:29500
--ep-size 8 --context-length 16384 --mem-fraction-static 0.85
--max-running-requests 1 --trust-remote-code
--dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv
--speculative-algorithm DFLASH --speculative-draft-model-path <DFlash2>
--speculative-num-draft-tokens 8 --speculative-draft-attention-backend triton
```

Reproduce: `snapshots/arm-20260902-060911.sh` (frozen), driven by
`submit.sh <label> <arm>`.

## Configuration facts, dearly bought

1. **`--ep-size 8` is mandatory.** Without it sglang tensor-shards the NVFP4
   MoE weights through their *packed* dimension and dies in the loader with
   shape mismatches at exactly half the expected width (3072 vs 6144).
2. **`fa4` draft attention hangs on SM90.** The card specifies FA4. It starts,
   captures graphs and serves, then deadlocks inside `libcuda` on the *first
   generate request*; sglang's 300 s scheduler watchdog kills the server. Both
   `triton` and `flashinfer` work and agree to 0.30%. **This is why the card's
   exact configuration cannot be run on Hopper**, and it is a plausible reason
   nobody has reported these numbers on GH200.
3. **`flashmla_sparse_q8` requires `--kv-cache-dtype fp8_e4m3`.** Plain
   `flashmla_sparse` avoids quantising the target's KV, which is what a
   numerics-sensitive measurement wants.
4. **The DSA prefill backend auto-resolves to `fa3` on Hopper**, which has no
   aarch64 build; this is unrelated to
   `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` and must be set explicitly.
5. **`SGLANG_CACHE_DIR` does not cover the `sglang.kernels` JIT cache.** Its
   resolver hardcodes `~/.cache/sglang/jit`
   (`kernels/jit/utils/compile/cache.py:302`) instead of deriving from
   `SGLANG_CACHE_DIR` the way `SGLANG_DG_CACHE_DIR` does. 6.0 GB had
   accumulated in `$HOME`. `SGLANG_JIT_CACHE_DIR` is now pinned to fscratch.

## Corrections to the sglang worklog

`jupiter-glm53-sglang-worklog/HANDOFF.md` recorded DFlash2 as **blocked**:
the draft checkpoint declares `architectures: ["DFlash2DraftModel"]` while
`speculative/dflash_utils.py:477` accepts only
`{"DFlashDraftModel", "Qwen3DSparkModel"}`.

**That was a misread.** Line 477 sits inside `is_nemotron_35_draft_config`, a
detector for a structurally distinct draft family; returning False for our
checkpoint is correct and harmless. `srt/models/dflash.py:1066` defines
`class DFlash2DraftModel(DFlashDraftModel)` and `EntryClass` (`:1146-1151`)
registers it - in that checkout as well as in current main. The arm was
descoped for nothing. Corrected there in `587ad95`.

## Multi-node topology: every alternative decomposition is closed

Attention at TP8 spans the node boundary, so every `o_proj` RowParallel
all-reduce crosses InfiniBand. Six arms were spent trying to keep attention
node-local. All are refuted, by the source or by measurement:

| approach | outcome |
| --- | --- |
| `--tp-size 4 --ep-size 8` | rejected: `parallel_hook.py:107` asserts `ep_size * moe_dp_size == tp_size` |
| TP4 + DP2 replication | 433 GB / 4 = ~108 GB/GPU of weights against 95 GB HBM |
| DP attention, `--dp-size 8` | OOM: replicates all 64 heads **and** the full KV per rank |
| DP attention, `--dp-size 2` | resolves back to `attn_tp_size=8` (the flag's help text says dp size must equal tp size), then OOM |
| `--pp-size 2` | `AssertionError: Pipeline parallelism is not compatible with overlap schedule, speculative decoding` - applies to **all** speculative arms, not just DFLASH |
| any of the above with DFlash2 | `speculative_hook.py:195,202` reject `enable_dp_attention` and `pp_size > 1` unconditionally |

Evidence that this mattered less than it appeared: **DFlash2 and MTP7 have
identical 77.5 ms step times despite DFlash2 verifying a wider block.** If
inter-node attention comms dominated, the wider verify should have cost
measurably more. Combined with decode being launch-bound on this stack (737
kernels/step), the limiter looks like per-collective latency and MoE traffic
(98.3% of weights) rather than attention bandwidth.

The three DFLASH restrictions are all worded *"Currently DFLASH ... only
supports"* - implementation gaps in a young feature, not fundamentals. A
measured win from lifting them would be a concrete upstream contribution; we do
not have that number, because PP is closed to speculative decoding generally
and DP attention does not fit in memory.

## Startup cost, for whoever runs this next

sglang takes ~19-31 min to serve where the fork takes 9m29s. The difference is
one line:

| | fork | sglang |
| --- | ---: | ---: |
| weight load | 61 s | 127 s |
| torch.compile | 82 s | - |
| **graph capture** | **3 s** | **844-918 s** |

The fork's launcher pins `cudagraph_capture_sizes: [8]` - one shape, because it
knows it serves batch-1 with an 8-token verify block. sglang generates a
production shape ladder (42 prefill buckets to 2048, 67 decode sizes to 512)
before it knows only one will be used.

Three arms measured the capture cost directly:

| arm | prefill shapes | prefill capture | target_verify | total |
| --- | ---: | ---: | ---: | ---: |
| baseline | 42 | 873.75 s | 22.42 s | 918 s |
| `--cuda-graph-max-bs-prefill 8` | 2 | 812.01 s | 23.28 s | 858 s |
| `--cuda-graph-backend-prefill disabled` | 0 | **0.00 s** | **821.31 s** | 844 s |

**Cutting shapes by 95% cut time by 7%, and disabling prefill capture moved the
cost wholesale onto `target_verify`.** So it is a one-time initialisation cost
(kernel JIT, autotune, workspace setup) that attaches to whichever graph
captures first - not per-shape work. Every capture-shape flag is therefore a
dead end.

Untested, and the remaining candidate: pin the capture list to the single shape
actually used (as the fork does) **and** run with the JIT cache warm on
fscratch now that `SGLANG_JIT_CACHE_DIR` is fixed. That combination has never
been run.

A source-reading subagent predicted the capture cost was linear in bucket count
(~20.8 s each, so ~62 s for 3 buckets). Its reasoning from the source was
careful and its conclusion was wrong; only measurement separated them. Its
`SGLANG_JIT_CACHE_DIR` finding was correct and valuable.

## Next

1. **Diff the fork's DFlash2 against `srt/models/dflash.py` and
   `srt/speculative/dflash_*.py`, targeting candidate generation.** Phase 49f
   put the whole deficit at position-0 recall; sglang is the working reference.
   This is the only work that can move the number.
2. sglang logs only aggregate `accept len`, not per-position rates, so a
   like-for-like per-position comparison needs either a patch to sglang or a
   recall probe equivalent to the fork's `VLLM_DFLASH2_RECALL_LOG`.
3. Not worth further arms: multi-node topology (closed above), FA3 for aarch64
   (throughput only, cannot move acceptance; the build is 4 config bugs fixed
   and ~50/396 objects compiled if ever wanted, at `sgl-kernel-fa3-build/`).
