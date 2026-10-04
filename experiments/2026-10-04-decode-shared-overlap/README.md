# Decode: shared expert vs routed MoE, and where the decode MoE time goes (2026-10-04)

## Shared expert in the served decode step (overlap_trace.py, trace-pdcp2, 4 ranks x 2,980 layers)

GLM-5.3's shared expert is TP-sharded (MergedColumnParallel gate_up 1024 wide
per rank + RowParallel down, reduce_results=False; ~19 MB bf16 per rank per
layer), summed into the routed output before the one post-MoE all-reduce.
vLLM runs it on an aux stream for <= 256 tokens, forked before the router GEMM
and joined at the shared+routed add. Median layer (us from router start):
router 0-4, top-k 4-9, route_prep 7-13, w13 9-121 (act's 64 CTAs PDL-resident
14-126), w2 122-181, add 183, AR+norm 185. Aux: gate_up 1-10 (over router /
top-k / route_prep), split-K reduce 13-19 and silu 20-25 (inside w13, no smem:
co-resident), down GEMM launchable at ~24 but starts at ~50 (waits 26-35 us,
p90 45-51: needs whole SMs; nvjet CTA = 384 thr x 168 regs = whole register
file + 164 KB smem; decode MoE CTA = 215 KB smem) and runs 50-61 inside w13
on SMs whose CTAs retired. Join never waits on the aux stream (slack median
119 us, p10 43). Bench (probe.py, one GPU, 9 hot + 2 cold): shared expert
adds ~9 us per layer on the side stream vs 21 us alone; in serving it is
hidden.

## Per-CTA trace of the served decode MoE (TD_CTA_TRACE, cta_analysis.py)

tiered_decode built with VLLM_TIERED_DECODE_DEFINES=TD_CTA_TRACE (each w13 /
w2 CTA logs SM, entry, ready, exit in %globaltimer plus the call's hot / cold
counts); dumped at /stop_profile by a local, uncommitted gpu_worker hook
(TD_TRACE_DIR). prof100k.sh ctatrace (20K new on 14K cached, then the decode
of the profiled request), ~990 launches per phase per rank:

| | w13 | w2 |
|---|---|---|
| experts per GPU per layer, median (p10-p90) | 7 hot (4-12), 3-4 cold (2-6) | same |
| hot CTAs done | ~43 us (~2.4 TB/s HBM) | ~24 us (~2.2 TB/s) |
| cold CTAs done | ~142 us (~387 GB/s C2C) | ~68-74 us (~385 GB/s) |
| cold tier ends the kernel | 93-95% | 88-92% |
| cold-only tail | median ~85 us (p90 ~170) | ~42 us (p90 ~85) |
| SM time idle in the kernel window | ~57% | ~56% |

Decode MoE is C2C-bound: the cold stream runs at the link's ceiling while
~110 SMs idle after the hot tier finishes. Fusing the shared expert or the
router into a persistent kernel would only fill idle SMs. The link is idle
outside the MoE phases (attention, dense GEMMs, all-reduce: ~45% of the step);
one correctly prefetched cold expert per layer would save ~55 us per layer
(~4 ms per step). Next: hit rate of a cold-expert predictor (previous step's
set, early router) on the agentic routing capture.

## Correction: the traced run was out of distribution (hot_vs_random.py)

The per-CTA trace above decoded raw vLLM-source completions (prof100k.sh).
There, 3.6 of ~11 distinct experts per GPU per layer per step were cold
(32%), the same as a random hot set of the served size (3,180 per GPU, 66%:
expected 32.7%). On held-out agentic steps the served profile gives 1.61
cold per GPU per layer (15.5%) vs 3.39 random, on Claude Code traffic 1.67
(16.8%) vs 3.26: the profile halves cold experts on real traffic. At ~1.6
cold the cold stream (~90 us) and hot stream (~80 us) are about balanced,
so the C2C-bound picture and the ~4 ms/step prefetch estimate above do not
carry over. Needs a trace on agentic traffic.

## Routing capture on the production config (vllm d08cd57e17)

Routed-expert return existed only in the V1 runner, and the scheduler
refused DCP (prompt routing is looked up by KV slot). Ported to the V2
runner (same design as upstream #50721) and made DCP-safe (prompt rows from
the step's own rows). claude-glm53-df2-dcp4-cap.sh = production launcher +
capture. Tested on the production config: traces [rows, 78, 8] in whole
8-row verify steps, 139-159 tok/s. A first test hung after compile: a
server killed mid JIT build had left torch_extensions/vllm_tiered_prefill/
lock (py-spy: all workers in file_baton.wait).

## First live capture on the production config (job 2173771)

59 requests / 28,698 verify steps from a Claude Code session, kept at
fscratch/routes-datasets/glm53-cc-20261004-job2173771 (README there).
hot_vs_random.py, served hot set (3,180 per GPU) vs random of the same size,
per GPU per layer per step: this session 2.00 cold of 10.34 active (19.3%)
vs random 3.40 (32.8%); agentic held-out 1.61 (15.5%), older Claude Code
1.67 (16.8%). The profile cuts cold reads 41% on today's traffic (52% on
its own data). Next sessions also log exact step times (steps.csv) and
request timing (vllm 37e14dfd45).
