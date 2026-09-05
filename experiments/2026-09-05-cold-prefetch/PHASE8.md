# Phase 8 — enabled in production, and measured there

Commit `067388f` (launcher), job `1679047` (traced pair), job `1673979` arm 1
(boot test). Production is `GLM-5.3-W4A16` served by
`experiments/2026-09-04-glm53-c1-df2/server.sbatch` via
`claude-glm53-c1-df2.sh`: DFlash2 drafter at 7 speculative tokens,
`max_num_seqs 1`, DCP 1, util 0.90, `max_model_len` 400000, reserve 10 GiB,
placement `glm53-w4a16-2496.json`, prefix caching on.

Phase 7 enabled the flag in the GLM-5.2 AutoRound launcher, which production
does not use. This phase puts it where production reads it.

## The boot test came first

The planner probe prices the slot off-node, but it cannot prove the server
starts. This launcher's own header records that reserve 6 and util 0.94 both
fail the runtime reserve check, because the dense drafter's KV cache is not in
`fixed_hbm_allocations` — roughly 2.18 GiB the planner never sees. Adding an
851 MiB slot to a configuration already described as tight is not something to
infer.

Job `1673979` arm 1, prod's config verbatim with the flag on and no profiler:

```
cold staging: 851 MiB budgeted per rank (largest cold tier 851 MiB)
reserve: 10.03 GiB free (minimum 8.38 GiB)
Tiered MoE residency: 2224 -> 2182 hot experts per rank
74 from slot, 1 from Grace          (all 56 chunks of the 96K prompt)
```

1.65 GiB of margin remains. The slot is **budgeted**, not taken from the
reserve: it is paid for by demoting 42 hot experts (1.9% of residency). The
one Grace fallback per chunk is the first MoE layer, which has nothing staged
behind it — the predicted steady state, unchanged from phase 3.

## Methodology: hot Marlin is the control

`stage(L+1)` is issued after `apply_tiered` for layer L and joined at L+1's
MoE. The copy is therefore in flight during the post-MoE all-reduce and during
layer L+1's attention, indexer and dense mm — but **never during hot or cold
Marlin**, which run after the join and before the next `stage`.

That makes hot Marlin the copy-free role, and so the control for any
measurement of this feature. It is what separates a valid pair from an
invalid one:

| pair | hot w13 | hot w2 | verdict |
| --- | --- | --- | --- |
| 1665068 / 1669308 (cross-job, cross-commit) | +5.5% | +6.0% | confounded |
| **1679047 (one allocation, one node)** | **-0.2%** | **+0.3%** | **clean** |

Call counts are identical between the arms here (78 / 157 / 21 / 74), which
also settles the `top_k_per_row_prefill` 81-vs-88 discrepancy seen in the
cross-commit pair: that was drift, not the prefetch changing call counts.

Both arms are cut to their first six chunks (`--max-chunks 6`; the staged
capture caught seven). `sparse_attn_indexer` climbs in lockstep across the
six — 50.27/70.01/85.97/102.20/108.46/131.98 against
50.28/71.40/86.67/103.00/108.21/133.43 — so the arms are matched on context,
not merely on count.

## Result

Per 8185-token prefill chunk, rank 0, mean of six matched chunks:

| | baseline | staged | |
| --- | --- | --- | --- |
| prefill wall | 1901.3 ms | **1581.6 ms** | **-16.8%** |
| GPU busy | 1964.7 ms | 1632.4 ms | -16.9% |
| cold Marlin | 604.6 ms | **236.2 ms** | **-60.9%** |
| hot Marlin | 190.6 ms | 190.5 ms | **-0.05%** (control) |
| all-reduce, rank mean | 174.1 ms | 144.3 ms | -17.1% |
| `sparse_decode_fwd` | 549.8 ms | 583.5 ms | **+6.1%** |
| `sparse_attn_indexer` | 91.5 ms | 92.2 ms | +0.7% |
| `aten::copy_` | 23.4 ms | 170.7 ms | +147.3 ms |
| `cum/busy` | 1.000 | 1.089 | overlap appears |

Roofline, with the cold tier priced against C2C when it reads Grace and
against HBM once staged:

| | effBW w13 / w2 | per-expert |
| --- | --- | --- |
| hot, baseline | 350 / 371 GB/s | 85.7 us |
| hot, staged | 345 / 363 GB/s | 87.3 us |
| cold, baseline | 124 / 143 GB/s | 234.7 us |
| cold, **staged** | **333 / 352 GB/s** | **90.2 us** |

**The cold tier goes from 2.74x the hot tier's per-expert cost to 1.03x.** It
is now reading at the hot tier's rate, which was the whole point.

## What it costs

`sparse_decode_fwd` +33.7 ms is the real price, and it is the copy contending
for bandwidth during attention. The ratio reproduces across checkpoints:

| | copy ms | attention delta | ratio |
| --- | --- | --- | --- |
| GLM-5.2 AutoRound W4G64 | 139.0 | +25.9 ms | 0.19 |
| GLM-5.3 W4A16 (prod) | 170.7 | +33.7 ms | 0.20 |

Two different quantisers, residencies and drafters give the same ~20%. Not
proof, but it is what "attention pays about a fifth of the copy's duration"
predicts, and it implies the cost **scales with cold-tier size** — worth
remembering if the placement ever pushes more experts cold.

Net: pay ~34 ms of attention contention to save ~368 ms of cold Marlin.

## All-reduce: total wait falls, imbalance does not

| | r0 | r1 | r2 | r3 | mean | spread |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 181.9 | 151.4 | 181.0 | 182.1 | 174.1 | 30.7 |
| staged | 167.3 | 118.2 | 143.7 | 147.8 | 144.3 | 49.1 |

The mean falls 17.1%, but the rank-to-rank spread **grows** (30.7 -> 49.1).
Rank 1 has the lowest all-reduce in both arms, meaning it arrives last in
both: the prefetch does not change who the straggler is here. GPU busy per
chunk is uniform across ranks in both arms (1964.7-1965.2, 1632.4-1633.1), so
all-reduce is the elastic term absorbing arrival differences, as before.

This is the second checkpoint showing the same pattern (phase 7 saw it on
GLM-5.2), so it is a reproducible property rather than run noise. Saying
"arrival skew shrinks" would be wrong in both cases.

What is **not** explained: the per-rank distribution does not follow the
per-rank staging load. Slots are 851 MiB on ranks 0/2/3 and 830 MiB on rank 1,
yet rank 0 carries a full slot and has the *fastest* attention in the staged
arm (583.5 against ~595 on the others) and correspondingly the highest
all-reduce. Slot size does not predict the ordering. Recorded as unexplained.

## Scope of the number

The -16.8% is **prefill on a fresh 96K context with prefix caching off**, which
is what the traced arms run so that a repeated prompt cannot skip the work
being measured. Production runs prefix caching **on**. A live Claude Code turn
re-sends a long, mostly unchanged prompt, so its prefill covers only the new
tail — one or two chunks, sometimes below the 1024-token gate. Per-turn saving
is therefore a fraction of this figure. Read -16.8% as the **upper bound per
fresh long context**, not as a per-turn speedup.

## Agreement with phase 5

Phase 5 measured -16.1% end to end (23.36 -> 19.60 s on the 96K prompt, three
reps per arm, both arms in one allocation) under **MTP3**. This phase measures
-16.8% per chunk from traces under **DFlash2**, a different job on a different
node. Two speculators, two methods, two allocations, 0.7 points apart.

That agreement is the reproducibility statement, and it is stronger than
either measurement alone.

## Files

| file | what |
| --- | --- |
| `job-prod53.sh` | boot test plus the traced pair, one allocation |
| `analyze_prod53.sh` | drives the pair; matches chunk counts, reads per-arm residency |
| `prod53-{baseline,staged}-table-1679047.txt` | full role tables |
| `prod53-{baseline,staged}-eff-1679047.txt` | roofline |
| `prod53-{baseline,staged}-roles-1679047.json` | machine-readable roles |
