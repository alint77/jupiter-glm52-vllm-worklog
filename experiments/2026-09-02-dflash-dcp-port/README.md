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

### DCP1 — the port is clean, and B is not a regression

| arm | AL mean | median | tok/s | Phase 52 baseline | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bc-eager-dcp1` | **5.7675** | 5.862 | 117.6 | 5.7190 | **+0.85%** |
| `bc-graph-dcp1` | **3.9699** | 4.0329 | 90.8 | 3.9626 | +0.18% |

Eager came in slightly *above* baseline (and 117.6 vs 114.1 tok/s), which is
the direction expected if the duplicate-slot write race commit B removed was
costing anything; it is small enough to be noise either way. What matters is
the negative result: **neither B nor C regressed CP1**, which the gates could
not show for B, since B is present on both sides of the `gate_cp1` diff.

The graph arm reproduces Phase 52 to 0.18%, so the 44% graph-vs-eager gap is
untouched by any of A/B/C — as predicted once `prepare_dflash_inputs` was
found to run outside the captured region.

### DCP4 — blocked below the port, in FlashAttention

Both prod arms died at engine init, eager and graph alike:

```
RuntimeError: scheduler_metadata must have shape (metadata_size)
  qwen3_dflash.py:647  in forward
  flash_attn.py:924    in forward
  flash_attn.py:1223   in _forward_with_dcp
  flash_attn_interface.py:347 in flash_attn_varlen_func
```

Read this carefully, because it is a *good* sign for the port: the draft is
now reaching `_forward_with_dcp` at all. Before commit C it could not run
under DCP; the slot math is being exercised. The failure is one layer down,
in the FA backend's DCP context path, which rejects the draft's shape.

The distinguishing feature is almost certainly the draft's multi-token query.
The same DCP4 topology runs in production under MTP
(`2026-08-29-glm53-c4/server.sbatch`), where `max_query_len == 1`; the DFlash
draft presents `max_query_len == 8`, which is what gates
`split_dcp_context_queries` at `flash_attn.py:574` and feeds
`should_split_fa2_dcp_context_attention` at 1170. The `scheduler_metadata`
built at 601 (`batch_size=num_reqs`, `causal=False`,
`seqlens=dcp_context_kv_lens`) is then handed to the plain varlen call at
1239 alongside `num_splits=attn_metadata.max_num_splits`.

**Not yet root-caused, and not attributable to this port.** Next session
should bisect by forcing the draft's attention backend away from FLASH_ATTN
(`flashinfer` and `triton` both worked in Phase 52's DCP1 arms) to establish
whether this is FA-specific, then decide whether the fix belongs in
`_forward_with_dcp` or in the draft's metadata build.

## Status

Port validated at DCP1 and shipped. DCP4 blocked on the FA issue above.
