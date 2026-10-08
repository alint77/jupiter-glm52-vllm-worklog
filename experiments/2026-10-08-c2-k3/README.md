# Concurrency 2 / 4: DFlash2 k=3 and MTP vs prod c=1 DFlash2 k=7

Quick probes rather than datasets: `conc_probe.py` streams chat requests over
~5K / ~50K tokens of this repo's source (prompts prefilled and cached first),
400 forced output tokens at temperature 1.0 / top_p 0.95, two reps. Decode
tok/s per request = (tokens - 1) / (last chunk - first chunk); aggregate =
all tokens / case wall time; acceptance length from the server's spec-decode
counters. Each run also takes torch-profiler windows (lone 5K request,
5K+5K pair, and 4 x 5K at c=4) before the timed reps. Prod stack otherwise
(reserve as noted, embedding on Grace, decode GEMM, frequency hot set).

    launch_mtp.sh <tag> <spec_k> <max_num_seqs> <reserve> <capture sizes> [--quad]
    run.sh <tag> [--profile-only]      # on a hold, via onnode.sh

The pool is max_num_seqs x 400K tokens (planner default): 400K / 800K / 1.6M.

## Startup

| config | reserve | KV pool | hot / rank | free at startup (min 2.53 GiB) |
|---|---|---|---|---|
| prod DFlash2 k=7, c=1 | 1.7 | 400K | 3,676 | 2.70 |
| DFlash2 k=3, c=2 | 1.7 | 800K | 3,596 | 2.71 |
| MTP3, c=1 | 1.7 | 400K | - | 1.98-2.06: refuses to start |
| MTP3, c=2 | 1.7 | 800K | - | 1.51-1.63: refuses to start |
| MTP1, c=4 | 1.7 | 1.6M | - | 1.67-1.74: refuses to start |
| MTP3, c=1 | 3.0 | 400K | 3,669 | 3.06-3.17 |
| MTP3, c=2 | 3.0 | 800K | 3,583 | 2.63-2.72 |
| MTP1, c=4 | 3.0 | 1.6M | 3,416 | 2.78-2.85 |

The planner under-charges the MTP layer (~0.5-0.9 GiB short at 1.7); at 3.0
MTP3 c=1 still keeps nearly prod's hot set, and c=1 has ~0.5 GiB to spare.

## Decode (tok/s; per request / aggregate; acceptance length)

| case | prod k7 c=1 | DF2 k3 c=2 | MTP3 c=1 | MTP3 c=2 | MTP1 c=4 |
|---|---|---|---|---|---|
| alone 5K | 169.6 / 160 | 150.0 / 142 | 161.3 / 153 (3.11) | 155.7 / 148 (3.01) | 111.8 / 108 (1.84) |
| alone 50K | 156.4 / 144 | 137.9 / 128 | 151.7 / 140 (2.93) | 142.7 / 132 (2.76) | 111.5 / 105 (1.86) |
| pair 5K+5K | (queued) / 155 | 129 / 175 | (queued) / 151 | 128 / 228 (2.98) | 98 / 181 |
| pair 50K+50K | (queued) / 161 | 120-127 / 207 | (queued) / 147 | 125-135 / 226 (3.03) | 97-100 / 180 |
| pair 5K+50K | (queued) / 166 | 121-133 / 231 | (queued) / 146 | 127-137 / 234 (3.11) | 94-101 / 183 |
| quad 4 x 5K | | | | | 81 / 297 |
| quad 4 x 50K | | | | | 76-78 / 279 |
| quad 2x5K+2x50K | | | | | 77-81 / 281 |

At c=1 the second request of a pair queues, so the per-request number is
the lone speed on another prompt (the 51K prompt accepts more).

## Profiles (rank 0, median step)

| window | period | GPU busy | idle |
|---|---|---|---|
| DF2 k3 c=2, alone (4 tok) | 24.85 | 18.15 | 6.70 |
| DF2 k3 c=2, pair (8 tok) | 23.25 | 21.88 | 1.37 |
| MTP3 c=1, alone (4 tok) | 19.12 | 18.00 | 1.12 |
| MTP3 c=2, alone (4 tok) | 19.20 | 18.23 | 0.97 |
| MTP3 c=2, pair (8 tok) | 22.82 | 21.88 | 0.93 |
| MTP1 c=4, alone (2 tok) | 16.53 | 15.89 | 0.64 |
| MTP1 c=4, pair (4 tok) | 19.63 | 19.00 | 0.63 |
| MTP1 c=4, quad (8 tok) | 22.06 | 21.49 | 0.57 |

Every window runs the one-kernel tiered MoE, `decode_gemm` and the skip-KV
stager (`_gather_rows`); the only Marlin MoE is the MTP layer's own (all-HBM,
0.10-0.18 ms/step). MTP's draft passes are in CUDA graphs (idle <= 1.1 ms).

The DFlash2 drafter ran eagerly then (~240-270 launches per step). Its large
profiled idle with a lone request at c=2 (6.7 ms) turned out to be the torch
profiler's per-launch cost, not serving overhead: un-profiled, the eager and
captured drafter give the same c=2 k=3 step time
(../2026-10-08-dflash2-cudagraph).

## Reading

- MTP3 c=2 and DF2 k3 c=2 are close. Lone requests are 8-9% below prod
  (155.7 vs 169.6 at 5K), pairs reach ~1.4-1.5x prod's aggregate at
  ~125-137 tok/s each. A rerun of DF2 k3 c=2 (../2026-10-08-dflash2-cudagraph,
  `run-c2k3-eager.log`) gave 155.6 lone 5K and 221-235 aggregate in pairs,
  i.e. MTP3's numbers; the first DF2 k3 run (150 / 175-231) was a noisy
  draw. MTP needs reserve 3.0 here, DF2 runs at 1.7 with ~13 more hot
  experts per rank.
- MTP1 c=4 tops out at ~280-300 tok/s aggregate with four in flight but
  only ~78-81 tok/s per request, and ~112 tok/s alone (acceptance 1.85 of 2):
  a throughput setting, not a latency one. It costs ~260 hot experts for
  the 1.6M pool.
- Reserve for MTP must be raised (3.0 here); 1.7 fails the startup check.
