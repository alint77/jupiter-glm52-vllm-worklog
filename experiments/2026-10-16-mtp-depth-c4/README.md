# MTP5 vs MTP3 at c=4 on the agentic timeline (2026-10-16)

Question from the mlx.fast Bonsai track (`../2026-10-16-cudafast-sidequest`):
MTP3's last draft position is accepted in 54.7% of rounds (live counters), so
the draft is censored. Would MTP5 collect more than its extra verify rows
cost? On paper about break-even: +26% tokens/round if acceptance holds at 0.8
per position, against about +23% step time at ~0.75 ms per row (c=4 M=16 -> 24).

Each arm runs the c4 launcher config (`arm.sh` = `../2026-10-11-nsys-fresh/arm.sh`
with SPEC_K and CAPTURE_SIZES overridable) on worktree pf5 (144f2f4aee), FCFS,
on its own fresh hold. The load is the 600 s c=4 closed-loop replay
(`timeline_save.py`: the c4-timeline driver plus saved outputs and /metrics
snapshots). Both arms capture 1-8, 12, 16, 18, 24, ... .

    run_queue.sh <hold>...          # queue/k3 k5 k5-r2 k3-r2; holds never cancelled
    ab_depth.py <arm dir>...        # tok/s, per-turn decode rate, acceptance per position
    lookup_glm.py <arm dir>         # prompt-lookup coverage on GLM's own outputs

Holds 2344299 (k3) and 2344298 (k5); results under
/e/fscratch/profound/naeimitabiei1/sliced-prof/mtpd-<arm>-<job>/.

## Result (2026-10-16, nodes jpbo-069-35 = hold 2344298, jpbo-069-36 = 2344299; MTP3 then MTP5 on each)

| arm | turns | output tok/s | per-turn decode tok/s (median) | TTFT mean | accept per position | tokens/round |
|---|--:|--:|--:|--:|---|--:|
| MTP3, node 35 | 157 | 227.5 | 102.0 | 2.92 | .759 .571 .435 | 2.77 |
| MTP3, node 36 | 159 | 224.0 | 103.4 | 2.78 | .764 .576 .433 | 2.77 |
| MTP5, node 35 | 155 | 240.7 | 98.1 | 2.59 | .745 .552 .414 .314 .247 | 3.27 |
| MTP5, node 36 | 150 | 242.8 | 97.6 | 2.43 | .738 .536 .395 .296 .229 | 3.19 |

Same-node per-turn decode rate on shared turns, MTP5 / MTP3: median 0.991 and
0.956, geomean 1.034 and 1.004 (MTP3 vs MTP3 across nodes: 1.010 / 1.015).
MTP5 collects +15-18% tokens per round, but each verify step costs about as much
more, so per-turn decode is flat to -4% at the median. Output tok/s is +6-8%,
but MTP5 also produced longer outputs and finished fewer turns (155/150 vs
157/159). No win: MTP3 stays. On this replay MTP3's last position accepts
in 43% of rounds, not 55% as on the live counters, so the censoring was
smaller than it looked.
