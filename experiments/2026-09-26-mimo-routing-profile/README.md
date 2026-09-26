# MiMo-V2.6 routed-expert placement, through the DAK lens

## Why

Reading the DAK paper (Lin et al., "Direct-Access-Enabled GPU Memory Offloading",
`nanoplm/agent_space/DAK_paper_tex`) raised two questions about our tiered MoE:

1. **Congestion control.** DAK reports that on GH200 more than ~8 SMs reading
   host memory, or too many in-flight requests per SM, drags HBM bandwidth down,
   and caps host-reading SMs. Our cold Marlin runs 1 CTA on every one of 132 SMs.
2. **Offload ratio.** DAK's effective-bandwidth model puts a memory-bound op's
   optimum where host and HBM reads take equal time, i.e. a host share of bytes
   of `B_c2c / (B_c2c + B_hbm)`. With our measured roofs (C2C ~373-400 GB/s for
   cold Marlin, hot Marlin ~2.15 TB/s, `2026-07-25-grace-bandwidth`) that is
   ~15%. MiMo runs **linear placement** (no routing profile exists), with
   3826 / 6624 experts hot per rank, so roughly 40% of routed bytes come from
   Grace: far past the balance point, i.e. C2C-bound.

## Question 1 is already answered by earlier measurements (no new run)

On Booster, GLM geometry, same Marlin kernel family and launch policy:

- **No HBM/C2C congestion at our operating point.** 3.1 TB/s of HBM streaming
  and 331 GB/s over C2C co-run with zero dilation once CTAs can co-reside
  (`2026-07-29-marlin-smem-monopoly`, Evidence 2); production layer unions sit
  within 2 µs of `max(hot, cold)` (`2026-08-01-marlin-tier-overlap`).
- **Capping cold SMs hurts Marlin.** Cold solo over C2C: 297.6 µs at 8 SMs vs
  117.3 µs at 32 SMs (`2026-07-27-green-context-marlin`). Marlin's cp.async
  pipeline has far fewer bytes in flight per SM than DAK's TMA windows, so it
  needs many SMs to fill the link and never over-drives it.
- **DAK's L2-bypass / read-amplification claim matches what we found
  independently**: Marlin re-reads an expert per token block, HBM absorbs the
  repeats in L2 and C2C does not (`2026-07-25-grace-bandwidth`). That is also
  why staging cold experts wins in prefill here (-17..-23% TTFT,
  `2026-09-22-mimo-v26-pro`), the regime DAK does not evaluate.

## Question 2: this experiment

Capture MiMo's routing under agentic coding, build a frequency placement
profile with the GLM pipeline, and measure cold share and decode speed.

Hypotheses, stated before the capture:

- **H1** under linear placement the held-out cold share of routed bytes is
  ~0.40 (the capacity cold fraction, 0.42, if routing were uniform).
- **H2** a frequency profile at the same 3826 slots/rank cuts it to <= 0.25,
  as GLM went 0.50 -> 0.23 at 50% residency.
- **H3** decode TPOT improves by several percent on the out-of-sample coding
  suite; GLM's profile+replicas gave -8% on the routed span.

## Pipeline

| file | role |
| --- | --- |
| `capture.sbatch` | production serve config + `--enable-return-routed-experts`, GLM agentic driver on the same node, `SHARD=0..3` |
| `tasks-{0..3}.json` | the 16 GLM capture tasks, split 4 ways |
| `mimo_profile.py` | runs the GLM `traces_to_manifest.py` / `optimize_routing_profile.py` with MiMo's layers (1..69) and 384 experts from the manifest |
| `step_tiers.py` | per 8-token decode step: distinct hot/cold experts per rank and layer, cold byte share, critical-rank cold count |
| `ab.sbatch` | same-node decode A/B, linear vs profile, server restarted per arm |

Validated on synthetic traces before the capture: manifest, optimizer,
`step_tiers.py`, and `load_tiered_moe_placement_profile` accepting the MiMo
profile (3826 hot slots on every rank).

## Interim result (94 traces, capture still running)

Traces are `(positions, 70, 8)`, layer 0 dense. They hold every *verified*
position, rejected drafts included (80 output tokens -> 144 rows = 18 DFlash
steps x 8), so `step_tiers.py`'s 8-row windows are real verify steps.

Split by task family (`--split-by domain`): 64 train requests, 30 held out
from task types the profile never saw; 98,144 routed positions. Frequency
residency at 3827 slots/rank (the runtime's count with the prefetch slot):

| held-out, per 8-token step | linear (today) | frequency profile |
| --- | ---: | ---: |
| routed cold-hit rate | 0.415 | **0.142** |
| distinct cold experts / layer / rank | 4.70 | **1.84** |
| distinct hot experts / layer / rank | 6.57 | 9.44 |
| cold share of expert bytes | 0.417 | **0.163** |
| critical-rank cold experts / step, summed over layers | 472 | 220 |

H1 holds (0.415) and H2 holds with margin (0.142 against <= 0.25). The profile
lands the Grace byte share at 0.163, on DAK's ~0.15 balance point.
`profile-interim-94.json` / `report-interim-94.json`.

## Full capture (609 traces)

Jobs 2071199-2071202, one hour each on four nodes: 609 traces, 920,888 routed
positions, 16 task families, no identity-fallback traces (max_tokens 8192).
Family split: 364 train, 245 held-out requests. `profile-3827.json`,
`report-3827.json`, `step-tiers-3827.json`.

| held-out, per 8-token step | linear | frequency profile |
| --- | ---: | ---: |
| routed cold-hit rate | 0.418 | **0.150** |
| distinct cold experts / layer / rank | 4.74 | **1.87** |
| distinct hot experts / layer / rank | 6.58 | 9.45 |
| cold share of expert bytes | 0.419 | **0.166** |
| critical-rank cold experts / step | 476 | 225 |

Same picture as the interim; the two hot sets overlap 0.939. Training-split
cold-hit is 0.119 against 0.150 held out, so some of the ranking is
family-specific, but most of it transfers.

## Decode A/B (interim profile)

Jobs 2071936-2071938, three nodes, four runs each in A-B-A-B / B-A-B-A order,
server restarted per run; the 16-prompt PyTorch coding suite (not in the
capture), 1024 tokens, greedy. `summarize_ab.py`:

| arm | n | TPOT | tokens / step | **step time** |
| --- | ---: | ---: | ---: | ---: |
| linear | 6 | 9.10 ms | 3.82 | **34.73 ms** (34.66-34.82) |
| profile | 6 | 6.78 ms | 3.70 | **25.03 ms** (24.91-25.20) |

**H3 holds, far above "several percent": -27.9% step time, -25.5% TPOT
(110 -> 147 tok/s).** Step time is tight to 0.3 ms within each arm on every
node; TPOT moves with DFlash acceptance, which drifts because hot and cold
Marlin can round differently under greedy decoding.

Consistent with the GLM tier-cost constants (~46 us per distinct cold expert,
cold-bound layers): 476 -> 225 critical cold experts per step is ~11.5 ms of
cold Marlin, against the 9.7 ms measured.

Residency caveat: with the profile the planner runs 3806 / 3867 hot slots on
the two rank types instead of 3827, because a frequency profile leaves some
layers with many more cold experts, which enlarges the prefill staging slot.
The planner then demotes or promotes arbitrary experts (it pops the highest
IDs, not the least used). That is ~1% of slots and inside the measured win;
ranking-aware promotion and demotion is a small follow-up.

## Status

- 2026-09-26: final-profile A/B with GSM8K 400 per arm (2073382, 2073383) and
  the full bench with the interim profile (2073120, TTFT at 32K/128K/240K)
  running.
