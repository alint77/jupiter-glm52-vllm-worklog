# DFlash2 drafter CUDA graph

Prod runs the DFlash2 drafter eagerly (`DFlash2Speculator.default_eager_draft
= True`): Phase 52 measured the captured FULL graph at 3.96 accepted tokens /
step against 5.72 eager on GSM8K (DCP1), undiagnosed. Eager costs ~240-270
individual launches per step: 1.89 ms span / 1.66 ms busy at c=1 k=7, and a
host-bound 3.54 ms span / 1.75 busy plus a 0.42 ms gap for a lone request at
c=2 k=3 (../2026-10-08-c2-k3).

## Audit

Checked and cleared by reading:
- the replayed attention metadata reads the same persistent buffers the
  capture used (`input_buffers.seq_lens` / `query_start_loc`, the block
  tables, `block_tables.slot_mappings`); FA's `scheduler_metadata` is copied
  into its persistent buffer with the capture-time `max_num_splits`;
- temperature, seeds and idx mapping are `copy_`'d into persistent buffers;
- the drafter capture runs inside its own `graph_capture()`, so the TP4
  candidate all-gather (`get_top_k_tokens` -> one-shot `all_gather`) takes the
  registered-address path, and `register_graph_buffers` runs at its exit;
- `prepare_dflash_inputs` and the context-KV precompute run outside the graph
  in both modes (as the 2026-09-02 divergence audit already found).

## Measured (prod c=1 k=7, DCP4; `run.sh`, `acc_probe.py`: 8 x 3K-token
## requests, 400 tokens, temperature 1.0, fixed seeds)

| arm | env | acceptance length | decode tok/s |
|---|---|---|---|
| eager (prod) | `VLLM_DFLASH2_EAGER_DRAFT=1` | 3.486 | 162.3 |
| graph | `=0` | 3.444 | 160.5 |
| graph, selector not compiled | `=0 VLLM_DFLASH2_NO_SELECTOR_COMPILE=1` | 3.457 | 162.9 |
| graph + probe | `=0 VLLM_DFLASH2_GRAPH_PROBE=25 VLLM_DFLASH2_RECALL_LOG=1` | 3.345 | 152.3 (probe re-runs) |

The probe replays the graph, snapshots its outputs, re-runs the same step
eagerly and diffs every stage. At all 8 probed steps (calls 25..200) the
draft hidden states (8 query rows), candidate sets, selector scores and
tokens are **bit-identical** (max |d| = 0, no NaN). The Phase 52 defect does
not exist on the current stack; acceptance differences above are trajectory
noise at temperature 1.0. It was most likely one of the fixes that landed
after Phase 52 for the DCP port (the `-1` `sample_idx_mapping` sentinel, the
group-dtype AOT schedule, the symmetrized window), but that is not bisected.

The probe now also diffs hidden states and selector scores, and fires every
N-th call up to 8 times (`dflash/speculator.py`, `dflash2/probe.py`).

## Served A/B (`chain_cg.sh`, 4 nodes x 4 arms alternating, `KINDS=cg0,cg1 compare_ba.py`)

cg0 = eager draft (old prod), cg1 = captured draft; prod `serve.sh` otherwise.

| | cg0 eager | cg1 captured |
|---|---|---|
| agentic decode | | **-0.088 +- 0.024 ms/step** (291 requests) |
| long-context decode (50K/130K) | | **-0.072 +- 0.027 ms/step** (64 requests) |
| GSM8K (8 x 200) | 0.909 | 0.914 |
| hot / free at startup | 3670 / 3.44-3.46 GiB | 3670 / 3.44-3.48 GiB |
| 388K stress | OK | OK |

Shipped: vllm 45d1b291ac (`default_eager_draft = False`; stale Phase 52
comments removed; `VLLM_DFLASH2_EAGER_DRAFT=1` restores eager).

## The profiled "host-bound eager drafter" was the profiler

`launch_c2.sh` (../2026-10-08-c2-k3 `run.sh --with-profile`):

| window | eager: period / idle | captured: period / idle |
|---|---|---|
| c=2 k=3, lone request | 26.62 / 8.45 ms | 18.40 / 0.68 ms |
| c=2 k=3, pair | 24.12 / 2.55 ms | 22.14 / 0.58 ms |
| c=1 k=7 | (prod dive 2: drafter 1.89 span / 1.66 busy) | 22.05 / 0.60 ms |

But without the profiler the two run at the same step time: acceptance /
tok/s gives 18.70 vs 18.70 ms (lone 5K) and 18.59 vs 18.63 ms (lone 50K) at
c=2 k=3, and the served A/B finds 0.09 ms at c=1. The torch profiler's
per-launch cost inflates the ~240-270 eager launches; the graph is one.
Profiled idle time around eager regions is not host overhead in serving.

Under the graph the drafter step is: eager prepare + context-KV precompute
(43-70 launches, ~0.6-0.8 ms span), the captured query forward (141 kernels,
~1.05 ms), then 43 eager launches (~0.12 ms).
