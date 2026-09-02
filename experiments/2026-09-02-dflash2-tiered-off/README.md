# Tiered-MoE-off control: does this fork's execution path cost DFlash2 its acceptance? (2026-09-02)

## Why this arm exists

Phase 49f localised the whole GSM8K deficit to one quantity -- the target's
token is absent from the draft's 16 candidates 36% of the time on math against
2% on code -- and showed that everything downstream of the candidate set is at
ceiling. It then named the target's precision as the untested variable, on the
inference that the card's 5.94 was measured against a BF16 target.

**That inference was wrong and is retracted.** The card's evaluation section
names its runtime but not its checkpoint, and incoai publishes the NVFP4 target
themselves. Verified on disk today:

| | |
| --- | --- |
| local `GLM-5.3-NVFP4` README references | `incoai/GLM-5.3-NVFP4`, `incoai/GLM-5.3-DFlash2` |
| local recorded commit | `54e52520606f96b3d9fc84088ad22882a61648ac` |
| HF API `sha` for `incoai/GLM-5.3-NVFP4` | `54e5252…`, `lastModified` 2026-08-28T15:03Z |

So Phases 43-49e have been running **incoai's own published pair, both at
current HEAD**, and getting 3.99 where the card reports 5.94. Quantisation of
the target is not the difference, and nothing needed downloading.

What remains between us and the card: SGLang vs vLLM, GB300 vs GH200, FA4 vs
FLASH_ATTN draft attention -- and **this fork's tiered MoE execution path**,
which is the only one of the four we own and the only one we can move. The
drafter conditions on the target's intermediate residual stream at layers
5/19/33/47/61/75; MTP reads only the final layer, and every quality gate in
this project (exact-text smoke, GSM8K score, KL against a same-fork control)
scores the target's *output*. A perturbation confined to the intermediate
states would be invisible to all of them and visible only here.

## Method

`arm-tieredoff.sh` runs the Phase 43 protocol unchanged -- incoai NVFP4 target,
incoai DFlash2 drafter, width 7, greedy drafting, T=1.0 / top-p 0.95, 64 GSM8K
samples at concurrency 1, 4096 max new tokens, V2 runner -- with
`--enable-tiered-moe` removed and the fork's pre-tiering native UVA expert
offload in its place (`--offload-backend uva --cpu-offload-params experts`).

Tiering off also lifts the tiered validator's pins (`vllm/config/vllm.py:2337`
and `:2333`), so `max_model_len` drops 400000 -> 16384: the eval never exceeds
~4.5K tokens, and the 21 GB MLA reservation is HBM this arm needs for weights.
Acceptance is unaffected by context capacity, and the arm is not a throughput
measurement -- UVA offload is expected to be slow.

Three arms submitted together (jobs 1618947, 1618948, 1618960), two offload
sizes bracketing the HBM fit rather than bisecting an OOM serially:

| arm | mode | `--cpu-offload-gb` |
| --- | --- | ---: |
| `tieredoff-dflash2-off40` | DFlash2 | 40 |
| `tieredoff-dflash2-off56` | DFlash2 | 56 |
| `tieredoff-mtp7-off40` | MTP7 control | 40 |

## Gate

Against the tiered numbers on the identical protocol:

| | tiered (49e/48) | this arm |
| --- | ---: | --- |
| DFlash2 GSM8K acceptance | 3.9951 | ? |
| DFlash2 GSM8K position-0 | 0.575-0.579 | ? |
| MTP7 GSM8K acceptance | 4.9348 | ? |
| MTP7 GSM8K position-0 | 0.899 | ? |

Readings:

- **DFlash2 recovers materially and MTP7 does not move** -> the tiered path
  perturbs the intermediate states the drafter reads, and the fault is ours.
  This is the only outcome that yields a fix.
- **Neither moves** -> the fork's MoE execution is exonerated, the last
  variable we own is closed, and the residual is SGLang/GB300/FA4 or the
  drafter's own behaviour against this target. DFlash2 should then be retired
  against MTP3 on acceptance, and the remaining open question is the
  throughput comparison, which acceptance cannot answer.
- **Both move together** -> the arm changed the harness, not the drafter;
  suspect `max_model_len` or the offload path and re-run with them matched.

Status: submitted 2026-09-02, results pending.

---

# Result: the tiered-off control is infeasible on one node (2026-09-02)

All three arms OOMed identically during startup. The traceback locates it
inside `get_offloader().wrap_modules(...)`: the native UVA path materialises
every weight on the GPU **before** wrapping it for host residency, so peak HBM
is the whole model regardless of `--cpu-offload-gb`. 433 GB of checkpoint
against 4x95 GB of HBM cannot survive that at any offload budget, which is why
offload 40 and 56 failed the same way and why the tiered path exists at all.

Retired, not retried. The control cannot be built on one node; building it on
two would mean porting the tiered/DCP work to multi-node, which is a project,
not an arm.

# What the inco.ai GLM-5.3 launch post establishes (2026-09-02)

Source: `https://inco.ai/blog/glm-5-3/`, 2026-08-28, read today.

**Their target was NVFP4**, confirming the retraction above: Figure 1 is
"inco.ai DFlash 2 NVFP4 Inco Engine 4.4x" against native FP8 autoregressive
decoding, and the NVFP4 checkpoint is described as being for "efficient
Blackwell inference".

**Two different measurements, two different stacks -- do not conflate them.**
The model card evaluates *acceptance* on SGLang / 4x GB300 / FA4. The blog
reports *end-to-end throughput* on Inco Engine, their proprietary stack.
Acceptance is engine-invariant given the same math, so the card's 5.94 is in
principle reproducible on any correct implementation; only the speedup figures
belong to Inco Engine.

New numbers the card does not carry, GSM8K at concurrency 1:

| | MTP | DFlash2 | ratio |
| --- | ---: | ---: | ---: |
| acceptance length | 5.12 | 5.94 | 1.160 |
| decode speedup vs autoregressive | 2.58x | 3.23x | 1.252 |

## The throughput question, stated honestly

Dividing those ratios gives 1.252 / 1.160 = **1.079**: on *their* stack,
DFlash2's draft-plus-verify cycle is about 8% cheaper than MTP's at equal
acceptance. It is tempting to carry that across and conclude DFlash2 loses
here, since our acceptance ratio is 3.995 / 4.935 = 0.81 and 0.81 x 1.079
= 0.87.

**That transplant is not valid and must not be quoted as a result.** The 8% is
a property of Inco Engine with FA4 on GB300, not of DFlash2; and our MTP7's
cycle cost carries this fork's tiered routed-MoE penalty, which Phase 42
measured as 65% proportional to token count. The two denominators are not the
same quantity.

What it supports is a **conditional**: *if* DFlash2's cycle-cost advantage on
this stack is no better than the 8% it achieves on theirs, DFlash2 lands ~13%
behind MTP7 end to end, and further behind the MTP3 production default. Since
DFlash2's whole case rests on a cheaper draft, that conditional is the thing to
falsify.

The way to get a real number is Phase 42's own cost model, `tok/s =
acceptance / step_time`, which fit its three points to 1.4% and already
contains MTP7's step time. It needs one DFlash2 step-time measurement on this
stack, and the W4A16 arms below log per-step timings. Take the number from
there rather than from anyone's blog.

## The Hopper NVFP4 path disfavours the tap-degradation theory

GH200 has no FP4 tensor cores, so the NVFP4 checkpoint runs through vLLM's
stock NVFP4 -> Marlin path (`tiered_moe_conversion.py:236`, delegating to
`convert_to_nvfp4_moe_kernel_format`). That path dequantises the fp4 weights
with their fp8 block scales into a **bf16 GEMM** and passes no activation
scales (`a13_scale=None`, `a2_scale=None`, i.e. weight-only). Native Blackwell
NVFP4 does FP4 tensor-core math including activations.

So our target's intermediate states are plausibly **closer** to full precision
than the ones the drafter was distilled against, not further. "Our numerics
degrade the taps the drafter reads" is now actively disfavoured rather than
merely unproven -- which is an argument against the whole family of
quantisation explanations, including the arm below.

# Arm: DFlash2 on the W4A16 g32 target (jobs 1619012, 1619013)

Harness is `arm-target.sh`, `arm-replicate.sh` with exactly three lines
changed (target path, cache tag, HBM reserve), so the numbers are comparable to
Phases 43-49e. Target `GLM-5.3-W4A16` int4 g32, profile
`glm53-w4a16-2496.json`, HBM reserve 6, GSM8K, greedy, width 7, 64 samples.

**What this arm can and cannot separate.** W4A16 g32 is *also* Marlin int4 --
the same execution path as the converted NVFP4, with different weights from a
different quantiser. So:

- **Acceptance flat** -> both the transcode theory and the
  quantiser-quality theory die together. This is the expected outcome given
  the precision argument above, and it closes the quantisation family.
- **Acceptance moves** -> ambiguous. It would not distinguish "the
  NVFP4 -> Marlin transcode hurt the taps" from "AutoRound g32 happens to
  preserve them better". A positive result needs a third arm to read.

Also to check from the server log before quoting any throughput from these
arms: the profile is c4-derived (2496 slots) but the arm runs c1/DCP1, and
Phase 46 established `VLLM_TIERED_MOE_PROFILE_CAP` is off by default, so
residency follows `available_hbm`, which differs at c1. Acceptance is
unaffected -- routing does not depend on which tier an expert sits in -- but a
throughput number taken at an unrecorded residency is not comparable to
anything.

| | tiered NVFP4 (49e/48) | W4A16 g32 |
| --- | ---: | --- |
| DFlash2 GSM8K acceptance | 3.9951 | pending |
| DFlash2 GSM8K position-0 | 0.575-0.579 | pending |
| DFlash2 step time | not measured | pending |
| MTP7 GSM8K acceptance | 4.9348 | pending |
| MTP7 GSM8K position-0 | 0.899 | pending |
| measured residency / profile cap | 2496 slots, cap off | pending |

# If the W4A16 arm is null: the one control that still separates stack from drafter

Every arm to date varies something inside this vLLM fork. The card's
acceptance protocol ran on **SGLang**, which is open, already checked out on
this cluster (`sglang-upstream`), and would run the same incoai pair on GH200.
That separates "this fork" from "the drafter's behaviour against this target"
in a way no in-fork arm can. The known aarch64 traps are recorded in
[[sglang-gh200-constraints]]: no FA3 build, so the DSA dense-prefill threshold
must be set to 0, and `PYTHONPATH` must shadow the venv's editable 5.2 fork.

Not launched. It is the next arm if W4A16 comes back flat.
