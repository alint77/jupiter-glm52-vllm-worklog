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

## Launch 1 and 2: OOM at 0.94 / reserve 6 (jobs 1658089, 1658174)

Both died identically, and **not** on anything this file got wrong about KV:

```
KV cache allocated, 400,064 tokens          <- exactly the planner's 6251 blocks
Tiered MoE residency: 2528 hot / 2272 cold  <- promoted from the profile's 2496
CUDA out of memory. Tried to allocate 2.00 GiB.
  GPU 0 ... 95.00 GiB total, 887.75 MiB free
```

Surfaced as `RuntimeError: torch_call_dispatcher("aten::new_empty", ...)`, which
is a wrapper; the OOM is the root cause. flashinfer's autotuner asks for 2.00 GiB
after the KV cache is up, and there was 0.88 GiB left.

The self-profiled KV was right. What was wrong was inheriting the MTP3
launcher's `0.94` / reserve `6` pair: the planner spends whatever utilisation
allows on hot experts (it *promoted* 2496 → 2528 to fill the budget), so the
headroom flashinfer needs was gone. DFlash2 carries six dense draft layers of
KV that MTP3 does not, so MTP3's headroom does not transfer.

Utilisation is the knob, and it shrinks the planner's budget without touching
KV. Two measurements bracket it:

| run | util / reserve | experts | KV | outcome |
|---|---|---|---|---|
| `repl2-eager-dcp1` | 0.90 / 7 | 45.8 GiB | 22.28 GiB | ran |
| jobs 1658089, 1658174 | 0.94 / 6 | 50.0 GiB | 21.92 GiB | OOM, 0.88 GiB free |

The 3.8 GiB difference is `0.04 x 95 GiB` — the planner tracks utilisation
almost exactly and spends it on experts. Reserve alone cannot substitute: 7 GB
at 0.94 fails the reserve check and refuses to start.

**Now set to 0.90 / 7**, repl2's pair, proven on this exact shape. Costs roughly
200 hot experts against 0.94.

Both knobs are overridable, so probing a tighter pair is one command:

```bash
CLAUDE_GLM53_GPU_UTIL=0.92 CLAUDE_GLM53_HBM_RESERVE_GB=6 ./claude-glm53-c1-df2.sh --start
```

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

Launched twice at 0.94 / reserve 6; both OOMed on flashinfer's autotuner.
Now set to 0.90 / 7 and **not yet re-launched**.
