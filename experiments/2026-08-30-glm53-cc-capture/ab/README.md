# Real-usage ranking A/B (2026-08-30): negative

Three pairs (jobs 1536849-51), shipped synthetic ranking against the one
re-derived from 140 real Claude Code traces, identical residency, ranking the
only variable.

| | tok/s | TPOT (ms) |
| --- | ---: | ---: |
| shipped (Phase 44 synthetic) | **202.76 ± 3.73** | 18.83 |
| real usage | **198.90 ± 3.98** | 19.10 |
| | **−1.90%** | |

Per pair: +0.58%, −5.55%, −0.63%. Against a ±2% spread this is no gain and
possibly a small loss, but it is emphatically not the win the placement metrics
predicted.

**Held-out cold-hit improved 24.6% (0.2791 → 0.2098) and throughput did not
follow.** That is the finding: at this operating point **cold-hit is not a proxy
for throughput**, and the whole capture-and-rerank pipeline optimises a quantity
that does not pay.

A mechanism consistent with the tier-cost work of the same day: cold-hit is
activation-weighted per token, while the cost is per *distinct expert per step*
and, under overlap, only the tier on the critical path is charged at all. A
ranking that reduces activation-weighted misses need not reduce the distinct
cold experts a step must stream.

Do not ship `glm53-w4a16-2496-realusage.json`. Before any further ranking work,
re-derive the objective against measured step cost — the placement optimiser's
`tail_objective` inherits the same activation-weighted assumption.
