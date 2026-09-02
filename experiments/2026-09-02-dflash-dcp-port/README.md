# Phase 53 arms — the #52188 DCP port

**Date:** 2026-09-02  **Jobs:** 1621605-1621608  **Branch:** `dflash2-backport`

The code change and its rationale live in
[2026-09-02-dflash-upstream-divergence](../2026-09-02-dflash-upstream-divergence/README.md).
This directory is the measurement.

## Why these four

The two on-GPU gates prove the *prepare kernel* is bit-identical at
`cp_size=1` and correct at `cp_size>1`, but they test that kernel in
isolation. Only an end-to-end arm shows commit B -- masking rejected suffix
rows to `PAD_SLOT_ID` -- did not move acceptance, since B deliberately changes
the kernel's output on any batch with rejections.

| arm | DCP | seqs | draft graph | compare against |
| --- | --- | --- | --- | --- |
| `bc-eager-dcp1`   | 1 | 1 | off | Phase 52 eager **5.7190** |
| `bc-graph-dcp1`   | 1 | 1 | on  | Phase 52 graph **3.9626** |
| `prod-eager-dcp4` | 4 | 4 | off | (new — previously refused) |
| `prod-graph-dcp4` | 4 | 4 | on  | (new — previously refused) |

Both prod arms run because the graph defect is still open and worth ~44%
acceptance, so the eager number is the one to quote for DCP4 until it is
fixed.

The draft graph is disabled the way Phase 52 did it: `VLLM_DFLASH2_PROBE`
set past the end of the run, which forces `need_eager` without the probe ever
firing.

## Protocol

Identical to Phase 51/52 so the numbers are comparable: GSM8K, 64 samples,
`replicate_dflash2_eval.py`, GLM-5.3-NVFP4 target with the GLM-5.3-DFlash2
drafter, `num_speculative_tokens: 7`, greedy draft sampling, one node, TP4.

```bash
bash agent_space/experiments/2026-09-02-dflash-dcp-port/submit.sh
```

`arm-dcp.sh` is `arm-replicate.sh` with `DCP` and `SEQS` lifted to env vars.
`submit.sh` freezes a snapshot per submission — editing a live arm script has
destroyed running arms before.

## Results

Pending.
