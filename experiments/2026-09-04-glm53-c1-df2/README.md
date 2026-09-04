# GLM-5.3 W4A16 + DFlash2, concurrency 1, DCP 1, 400K context

A serving launcher, not an experiment: `claude-glm53-c1-df2.sh` at the repo
root points here.

## Why this shape

The tiered-MoE production shape with nothing relaxed. All three shape pins in
`vllm/config/vllm.py` hold natively at c=1 — `max_num_seqs > 1` without DCP
(2330), `max_num_batched_tokens != 8192` (2342), `max_model_len != 400000`
(2346) — so `VLLM_TIERED_MOE_RELAX_SHAPE` is absent and the validator stays in
force. The c=4 sibling has to lift it.

DCP 1 is deliberate: DFlash2 loses ~38% of its acceptance under DCP4 and the
cause is still undiagnosed (3.5098 AL at DCP4 c=4 against 5.7046 at DCP1 c=4,
both from the 32K 2×2 diagnostic in `2026-09-02-dflash-dcp-port`).

At c=1 the context is free. The planner derives `num_blocks` from
`max_model_len` and `max_num_seqs` alone
(`tiered_moe_kv.py:309-312`), so a single sequence asks for
`ceil(400000/64) + 1 = 6251` blocks. That term scales with `max_num_seqs`
while DCP 1 leaves the MLA cache replicated rather than sharded, which is
exactly why the c=4 sibling cannot reach 400K.

## Anchor

`repl2-eager-dcp1` (`2026-09-02-dflash-dcp-port/submit-repl.sh:25`) ran this
shape — DFlash2, `DCP=1`, `SEQS=1`, run-server.sh's default 400K:

| | |
|---|---|
| acceptance length (mean) | 5.6265 |
| output throughput | 116.74 tok/s |
| available KV cache | 22.28 GiB |
| GPU KV cache size | 400,064 tokens |
| expert residency | 2317 hot / 2483 cold per rank (45.8 GiB) |

Single-stream, 64 GSM8K samples, DFlash2 card protocol (T=1.0, top_p=0.95).
**Not** comparable to the 209.3 tok/s aggregate figure in
`2026-09-02-df2-c4-perf`, which is MTP3 under real concurrent load.

That arm served **NVFP4, not W4A16**. KV sizing does not depend on the weight
quantisation, so the cache figures carry over. Expert residency does not —
W4A16 weights are smaller, so expect more than 2317 hot experts. The
`Tiered MoE residency` log line reports the actual number.

## Three OOMs, and the planner defect behind them

| job | util / reserve | experts | KV declared | free | result |
|---|---|---|---|---|---|
| 1658089, 1658174 | 0.94 / 6 | 50.0 GiB | 21.92 GiB | 0.88 GiB | OOM on a 2.00 GiB alloc |
| 1658456 | 0.90 / 7 | 49.1 GiB | 19.06 GiB | 1.80 GiB | OOM on a 2.00 GiB alloc |

All three allocated KV correctly — 400,064 tokens, the planner's 6251 blocks —
then died during KV cache init. The error arrives as
`RuntimeError: torch_call_dispatcher("aten::new_empty", ...)`, a wrapper; the
root cause is `CUDA out of memory` a few hundred lines earlier. flashinfer's
autotuner is *not* the culprit — it logs "Autotuning process ends" for all four
workers before the failure.

### The mechanism

The planner budgets HBM from `fixed_hbm_allocations`, which holds
`main_mla_cache` (`tiered_moe_physical.py:227`) and `indexer_cache` (`:222`) and
nothing else. The cache it then allocates is sized by
`get_tiered_kv_available_memory` (`tiered_moe_kv.py:140`, via
`kv_cache_utils.py:2201`), which sums main **and indexer and draft** specs. Per
rank at 6251 blocks:

| component | bytes | in the planner's budget? |
|---|---|---|
| `main_mla_cache` | 19.06 GiB | yes |
| `indexer_cache` | 1.03 GiB | yes |
| draft cache (6 dense layers) | **2.18 GiB** | **no** |

The drafter's layers are invisible to the planner, which spends that 2.18 GiB
on hot experts and then cannot fit the cache it just promised. MTP3 never trips
this: its grafted layer is an MLA layer counted in `main_specs`, so
`main_cache_bytes` already covers it — which is why the MTP3 launcher runs at
reserve 6 and this one cannot.

### Which knob actually moves memory

Reserve, 1:1. Measured: 6 → 7 GB took experts 50.0 → 49.1 GiB and free
0.88 → 1.80 GiB.

Utilisation does **not**. 0.94 → 0.90 changed only the *declared* KV budget
(21.92 → 19.06 GiB), which is slack the tiered path ignores because the planner
forces the block count regardless — both runs allocated the identical 400,064
tokens. An earlier revision of this file credited utilisation for the gain;
that was wrong, and the numbers above are why. Keep 0.90 anyway: it is the
precondition for a reserve above 7, which at 0.94 fails the reserve check.

**Now 0.90 / reserve 10** — the unbudgeted 2.18 GiB plus margin. Predicts
roughly 4.8 GiB free.

```bash
CLAUDE_GLM53_HBM_RESERVE_GB=9 ./claude-glm53-c1-df2.sh --start   # tighter probe
```

### Follow-up

The real fix is to add the draft specs to `fixed_hbm_allocations` so the
planner stops handing away memory it has already committed. Until then every
dense-drafter shape (DFlash, DSpark) needs a reserve inflated by its own draft
cache. Worth a patch; it is a genuine accounting bug, not a tuning quirk.

## No `--kv-cache-memory`, deliberately

Every other W4A16 config here passes `21689598771` (which buys 400,064 tokens
per rank under MTP3). This one passes nothing.

The flag only short-circuits the profiling run (`gpu_worker.py:471`); it is not
an input to the tiered planner. DFlash2 charges six dense draft layers of KV
per block where MTP3 charges one grafted MLA layer, so the MTP3 constant is
wrong here — and deriving a replacement is what cost the c=4 launcher three
revisions. `repl2-eager-dcp1` passed no flag and profiled to a self-consistent
22.28 GiB.

Cost: ~3.5 min of startup profiling (repl2's log runs residency 14:39:02 →
KV 14:42:38), which the sbatch's 240 × 10s ready loop absorbs.

**Confirmed by launch:** jobs 1658089 and 1658174 both profiled to 21.92 GiB
and allocated exactly 400,064 tokens — the planner's 6251 blocks, predicted
before the run. The no-flag path works on W4A16. Those jobs died on HBM
headroom, not on KV sizing.

## Status

Three launches, three OOMs, all from the draft-cache accounting gap above.
Now 0.90 / reserve 10 and **not yet re-launched**.
