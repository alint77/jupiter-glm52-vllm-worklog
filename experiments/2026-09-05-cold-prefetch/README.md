# Cold-expert H2D prefetch for prefill (2026-09-05)

**Status: planned, nothing built.** See [PLAN.md](PLAN.md).

Hide the cold tier's Grace->HBM traffic behind layer compute during prefill so
cold Marlin reads from HBM at hot's rate. Sized from
[`2026-09-04-mtp3-profile`](../2026-09-04-mtp3-profile/PREFILL.md): worth up to
**-373 ms per 8192-token chunk (19.0%)**, ~-4.5 s on a 23.2 s TTFT at 96K.

Feasibility is measured, not modelled: across all 444 layer transitions in the
6 captured chunks, **none** fails to hide the next layer's cold operands, at
either 450 GB/s or the 373 GB/s measured rate; worst case is 7.04x margin at a
6.1% duty cycle.

## Contents

| file | what |
| --- | --- |
| `PLAN.md` | design, hooks, gating, phases, kill criteria, risks |
| `all64_tier_probe.py` | resolves whether the all-64-expert layers are hot or cold |
| `all64-tier-1665068.txt` | its output: they are **hot**, so no all-cold-64 layer exists |
| `PHASE1.md` | DMA microbenchmark: all three mechanisms reach 417-421 GB/s |
| `PHASE2.md` | in situ staging, verified, still executing from Grace |
| `PHASE3.md` | cold tier executes from the staged slot (weights **and** scales) |
| `PHASE4.md` | the planner budgets the slot; verified to precede the reserve check |
| `PHASE5.md` | paired GSM8K gate: p=0.888 over 1319 questions, prefill -16.1% |
| `PHASE6.md` | the same gate on the single-stream M=8192 path: p=1.000 over 1000 |

## Two facts that shaped the design

* The all-64-expert Marlin calls are **hot** (250/228 GB/s against a hot median
  of 268/243 and a cold median of 93/92), plus two degenerate launches doing
  864 MiB in 22 us with no routed rows. So the largest cold layer is 41 experts
  / 830 MiB and the double buffer is 1.62 GiB, not 2.53.
* Tier storage is **component-major**, so a slot sized for the largest layer
  cannot be sliced for a smaller one -- views are rebuilt per (layer, slot).
