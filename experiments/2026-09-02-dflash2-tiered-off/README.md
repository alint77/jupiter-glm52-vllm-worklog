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
