# GLM cost table for the decode MoE balancer, and the hot-set order (2026-10-08)

The MoE all-reduce wait (1.33-1.42 ms/step in `../2026-10-07-decode-dive-2`)
is the MoE chain's spread across ranks. Question: does the replica balancer's
cost table (`COST_US` in tiered_decode.cu, measured on MiMo / MXFP4) misplace
work for GLM?

## Grid (`grid.sh`, node 2221619)

The kernel in this tree over the table's grid (hot 0-24 x cold 0-6), INT4 and
MXFP4, pools of 49 hot / 34 cold, NUMA-bound, 3 repeats with the format order
alternating: median repeat spread 0.3% (p90 1.5%; the 2026-10-04 bench drifted
~20%). INT4 / MXFP4 = 1.054 median (p10 1.029, p90 1.056), i.e. near-uniform.
Today's kernel is faster than the shipped table on hot-heavy cells ((16,1) 130
vs 155 us, (24,0) 172 vs 196).

`bench_fmt_distinct.py` is ../2026-09-27-glm53-mtp7-profile/bench_fmt.py with
distinct pool slots per call: the original draws slots with replacement, two
experts can share a slot, and the kernel then faults (illegal address) at
random cells. The planner's maps never share slots, so prod is not exposed.

## Replay (`replay.py`, live Claude Code capture, 3,241 steps, 3,676 hot/rank)

The balancer's assignment (profile_grid.assign) under each table, scored with
the measured INT4 cells:

| table | slowest rank | mean rank | spread |
|---|--:|--:|--:|
| shipped | 9.634 ms/step | 8.137 | 1.497 |
| GLM INT4 | 9.582 | 8.036 | 1.546 |

-0.05 ms/step: not worth a kernel change. The ~1.5 ms spread is structural
(hot experts fixed to their owner, cold counts in whole experts).

## Hot-set order (`replay_hotset.py`, held out)

Prod's profile lists 3,239 hot per rank in frequency order; the planner fills
the rest of the 3,676 slots in expert-id order ("no frequency information").
Ranked by route counts on the even capture files, evaluated on the odd (2,228
steps):

| hot set | slowest rank | mean rank | cold per rank-layer |
|---|--:|--:|--:|
| id-order promotion (prod) | 9.601 ms/step | 8.102 | 1.29 |
| frequency-ordered promotion | **8.706** | 7.320 | 0.90 |
| each rank's top 3,676 by frequency | 8.742 | 7.219 | 0.80 |

Predicted -0.9 ms/step of MoE time on the slowest rank. Next: a profile whose
lists extend in frequency order past the budget, and a served A/B.

## Served A/B (`chain_pf.sh`, 4 nodes x 4 arms alternating, `KINDS=pf0,pf1 compare_ba.py`)

`profiles/glm53-w4a16-agentic-3239-r2000-ccfreq3676.json`: prod's lists plus,
per rank, the 437 most-routed remaining experts over all 76 live-capture files,
appended in descending frequency. Owners, replicas and hashes unchanged.

| | prod profile | frequency-promoted |
|---|--:|--:|
| hot / rank | 3,676 | 3,670 |
| tightest-rank startup free | 2.69 GiB | 3.46 |
| agentic decode (287 requests) | | **-0.20 +- 0.03 ms/step** |
| 50K / 130K decode (64 requests) | | **-0.32 +- 0.03 ms/step** |
| GSM8K 200 (8 runs each) | 0.918 | 0.913 |

TTFT, 388K stress (16/16) and peak memory unchanged. Smaller than the replay's
-0.9: the hot set is ranked on live Claude Code routing, the bench replays the
MiMo-capture task set. The new profile leaves ~0.75 GiB more free with 6 fewer
hot experts (not investigated; reserve left at 1.7). serve.sh now defaults to it.
