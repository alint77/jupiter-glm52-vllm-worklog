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

**Unverified:** no W4A16 config has yet run without the flag, so the no-flag
path is proven on NVFP4 only. A profiling failure is loud and cheap to
diagnose; a wrong constant is not. The first launch confirms it.

## Status

Written, syntax-checked, **not yet launched**.
