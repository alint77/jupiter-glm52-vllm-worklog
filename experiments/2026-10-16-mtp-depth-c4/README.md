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

## SRPT paired rerun (same nodes, MTP3, `VLLM_SCHED_INTERLEAVE_SRPT=1`, queue items k3-srpt-a/b)

Each SRPT arm is compared with the FCFS MTP3 arm from the same node (`ab_depth.py`, `../2026-10-15-c4-timeline/ab_srpt.py`):

| node | arm | turns | output tok/s | TTFT follow-up mean / p90 | TTFT max | per-turn decode, SRPT / FCFS |
|---|---|--:|--:|---|--:|---|
| 35 | FCFS | 157 | 227.5 | 2.04 / 1.44 | 77.9 | |
| 35 | SRPT | 147 | 213.6 | 1.45 / 0.77 | 107.5 | median 0.987, geomean 0.937 |
| 36 | FCFS | 159 | 224.0 | 1.92 / 1.86 | 76.9 | |
| 36 | SRPT | 159 | 239.4 | 1.05 / 0.81 | 106.0 | median 1.002, geomean 0.944 |

As in the first pair (2026-10-15: 2.39 vs 2.85 s mean), SRPT cuts follow-up TTFT
consistently: mean -30 to -45%, p90 about halved. It costs the cold long turns
(worst TTFT 77 -> 107 s, both nodes) and ~6% geomean per-turn decode rate.
Throughput is -6% on one node and +7% on the other, within run-to-run spread.
A trade, not a free default.
