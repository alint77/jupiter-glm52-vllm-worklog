# Phase 4 — budget the staging slot through the planner

Commits `056f1893da`, `f6a7e8727f`. Verified in situ by job `1669612`.

## The hole

The slot was allocated lazily at the first prefill chunk. That is *after*
`validate_tiered_moe_observed_hbm_reserve` has measured the margin and signed
it off, so 830 MiB came silently out of the runtime reserve. The check would
pass on memory the slot was about to take.

## The fix, and why it needed a fixed point

`cold_staging` is now a derived allocation the planner reserves before
placement is decided. Budget and plan are mutually dependent: reserving the
slot shrinks the hot budget, which demotes experts into the cold tiers, which
grows the slot those tiers need. One pass under-budgets.

The sequence is monotone and bounded by a layer's full expert count, so it
terminates; it fails closed if it has not settled in eight passes. Off-node
against the real checkpoint it converged in three (`planner_staging_probe.py`).

The slot is then allocated **eagerly in the worker, before the reserve is
measured**, and the coordinator raises if what the placement needs exceeds what
was budgeted -- or if the prefetch is on with cold tiers registered and no
budget is visible at all, which is always a plumbing fault since the planner
and the runtime enable the slot from one shared predicate.

## Verified in situ, not just in tests

The ordering is the one property no unit test covers, because it is a fact
about worker startup:

```
04:19:39  cold prefetch: 75 layers, slot 830 MiB   (lines 86-89, all four ranks)
04:20:29  observed HBM reserve: 10.31 GiB free (minimum 8.38 GiB)   (lines 152-155)
```

Allocation first, by 50 seconds, on every rank. The reserve check now passes
*with the slot already resident*, which is the entire point.

| | baseline | staged |
| --- | --- | --- |
| residency | 2325 hot / 2475 cold | **2284 hot / 2516 cold** |
| `cold_staging` budgeted | -- | **830 MiB** |
| actual slot | -- | 830 MiB (810 on rank 1) |

The budget matches the slot exactly. The cost is **41 hot experts, 1.76% of
residency**. The off-node probe predicted 871 MiB and 43 experts from
approximate capacities; the real placement gives 830 and 41.

Both residency lines appear in the staged server's log -- 2325 from the fixed
point's first pass with no budget, then 2284 once it settles. That is the
iteration being visible, not two plans.

## Note

`build_tiered_moe_plan` references an undefined `vllm_config`
(`tiered_moe_plan.py:178`), so the plan-only CLI cannot price a scenario. That
is pre-existing and unrelated -- the file was last touched by `cec73c66b3` --
but it is why `planner_staging_probe.py` calls the scenario planner directly.
