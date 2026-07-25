# Hot-first tier reorder — qualification (reverted)

Qualification of the A4 finding from
[tier-overlap](../2026-07-25-tier-overlap/README.md). Job 1041016, MTP3 c4
DCP4, 7 GB reserve, against the post-draft-sync baseline.

Correctness passes: semantic completion and the exact-400K golden SHA
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`.

| Case | metric | draft-sync | tier-order | Δ |
| --- | --- | ---: | ---: | ---: |
| realistic c4 warmed | output tok/s | 185.05 | 184.42 | −0.34% |
| realistic c4 first use | output tok/s | 112.44 | 111.42 | −0.90% |
| 4K c4 r1 | output tok/s | 99.13 | 107.26 | +8.21% |
| 4K c4 r2 | output tok/s | 110.25 | 106.33 | −3.55% |
| 396K c4 | median ITL | 58.13 ms | 56.76 ms | −2.36% |

The 4K pair disagrees by 12 points in opposite directions, so that is variance.
The cleanest metric, the routed layer span from the trace, moves 25.869 →
25.627 ms (−0.94%).

The change was verified live rather than assumed: `hot_start − cold_start` flips
from +2.05 to −2.08 us median and hot starts first in 98.7% of layers, up from
2.0%. It did what it was meant to do and produced nothing, because production
co-residency was already 81.5%.

**Reverted.** See the tier-overlap report for the full refutation.
