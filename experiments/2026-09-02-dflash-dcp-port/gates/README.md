# FA3 scheduler_metadata: what its size actually depends on

Two throwaway-cheap scripts that run on one GPU in seconds, written to stop
the DCP4 investigation from guessing. Both call the real FA3 ops.

```bash
.venv/bin/python gates/fa3_meta.py      # what changes metadata_size
.venv/bin/python gates/fa3_mismatch.py  # which disagreement raises
```

## `fa3_meta.py` — the size formula, measured

`metadata_size` depends on **batch, `causal`, and `num_splits`. Nothing else.**

| varied | result |
| --- | --- |
| batch 1-4 / 5-8 / 16 | 9 / 17 / 33, i.e. `1 + round_up(batch,4)*2` |
| `causal=True` | 13 (vs 9) — **differs** |
| `num_splits=1` | 5 (vs 9) — **differs** |
| `num_splits=8` | 9 — same |
| `max_seqlen_q` 1/8/16, `max_seqlen_k` up to 400000 | all 9 — same |
| `num_heads_q` 4/8/32, `num_heads_kv` 8, `headdim` 64 | all 9 — same |

Note `flash_attn.py:404` documents the size as `1 + round_up(batch_size,4)*4`.
That is stale for this FA build -- the measured factor is 2, not 4. It only
over-allocates the persistent buffer, so it is not a bug, but do not trust the
comment as the formula.

**This immediately rules out `num_heads_q`.** The draft and target have
different head counts, which made a head-count mismatch the obvious suspect --
`schedule()` builds with `num_heads_q * dcp_world_size` -- but head count does
not enter the size at all. (`num_heads_q` is in any case taken from the layers
via `get_num_attention_heads_from_layers`, not from `model_config`, so the
draft's builder already gets the draft's own count.)

## `fa3_mismatch.py` — which disagreement produces the error

Reproduces `RuntimeError: scheduler_metadata must have shape (metadata_size)`
locally, by building metadata with one parameter set and calling with another:

| build vs call | outcome |
| --- | --- |
| everything agrees | OK |
| `causal` True vs False | **RAISES** |
| `causal` False vs True | **RAISES** |
| batch 8 vs 4 | **RAISES** |
| `num_splits` 1 vs 0 | **RAISES** |
| `num_splits` 0 vs 8 | OK |
| `num_heads_q` 32 vs 8 | OK |

So exactly three candidates remain for the DCP4 failure: **batch**, **causal**,
or **`num_splits` where one side is 1**.

## Why this needed an instrumented arm anyway

Tracing all three through the source says they agree:

- **batch** -- `schedule()` at `flash_attn.py:601` takes `batch_size=num_reqs`;
  the call at 1223 takes `cu_seqlens_q = attn_metadata.query_start_loc`
  (`flash_attn.py:919`), and the draft builds that as
  `query_start_loc[:num_reqs_padded+1]` against `num_reqs=num_reqs_padded`.
- **causal** -- both the DCP-branch build (line 606) and the context call
  (line 1233) hardcode `causal=False`, independent of the layer's own causal.
- **`num_splits`** -- built from the local `max_num_splits` (line 531), stored
  into the metadata unchanged (line 685), read back at the call (line 1244).

The source model therefore predicts no mismatch, and the failure disproves the
model rather than any one hypothesis -- which is why the next step is the
instrumented arm dumping all three at both ends, not more reading.
