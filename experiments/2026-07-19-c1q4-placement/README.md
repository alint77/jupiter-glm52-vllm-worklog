# MTP3 c1/q4 domain-aware expert placement

This experiment replaces the original synthetic, pre-MTP routing profile with
target-verification traces from MTP3 at concurrency one (q4). The workload is
balanced across Python, PyTorch, machine learning, and mathematics: four
training and two held-out prompts per domain, each with 256 forced output
tokens.

The comparison holds the 2,870-HBM-expert/rank budget fixed:

1. Existing synthetic/pre-MTP profile.
2. Domain-mixed MTP3 per-expert profile.
3. Layer-concentrated hybrid profiles. These reserve complete layers in HBM
   when eliminating the second tier is worth a configurable mixed-layer
   penalty, then spend the remaining slots on the most frequent experts.

Route capture is decode-only and retains all four target routes in every MTP
verification step, including rejected drafts. Normal routed-expert responses
remain unchanged unless the internal `return_rejected_routed_experts` request
flag is set.

The matched c1/q4 runs use the qualified V1 runner. Background job `977479`
showed that the V2 runner does not currently become ready at DCP1: it hangs
after compilation during graph warmup. The V2 result remains qualified for
DCP4/c4, not c1.

## Initial cold-run screening

All runs used c1, MTP3/q4, 2,870 HBM expert slots per EP rank, a 10 GiB
HBM reserve, 24 prompts, and 256 forced output tokens per prompt.

| Placement | Full / mixed / cold layers | End-to-end tok/s | Mean TPOT | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| Existing synthetic profile | 0 / 75 / 0 | 63.23 | 10.45 ms | 2.958 |
| MTP frequency profile | 0 / 75 / 0 | **69.48** | 9.23 ms | 2.901 |
| MTP tail-aware per-expert | 0 / 75 / 0 | 68.50 | **9.15 ms** | 2.909 |
| Layer-concentrated hybrid | 44 / 1 / 30 | 65.81 | 10.06 ms | 2.915 |

The frequency profile improved end-to-end output throughput by 9.9%. The
tail-aware profile improved steady decode throughput from 95.7 to 109.3 tok/s
and reduced TPOT by 12.4%. It was also fastest on three of four held-out
domains: Python 116.63, PyTorch 106.26, machine learning 104.53, and math
114.61 tok/s.

The layer-concentrated implementation worked and beat the old profile by 4.1%,
but it was 5.3% slower than the frequency profile. Eliminating nearly every
mixed-layer launch did not compensate for increasing the held-out cold-critical
route count from 259 to 348. The best c1/q4 direction is therefore the new
domain-trained per-expert placement, not whole-layer concentration at this HBM
budget.

The end-to-end column above includes first-use JIT/TTFT spikes and is retained
only as the initial screening result. It must not be compared with the old
129-136 tok/s exact-400K number, which was decode-only TPOT on one synthetic
request.

## Warmed realistic-prompt qualification

Performance decisions now use the 24-prompt Python, PyTorch, machine-learning,
and math suite as the primary gate. Each job disables prefix caching, runs one
complete excluded warmup over all prompt shapes, then saves two repetitions of
24 requests with 256 forced output tokens. Both end-to-end throughput and
steady decode (`1000 / mean TPOT`) are reported.

The source A/B uses the same `0 / 75 / 0` frequency placement. Commit
`0d87dd9ae` is the parent of the speculative route-capture change, and
`6490e584f` is the current source.

| Source | End-to-end tok/s | Mean TPOT | Decode tok/s | Mean TTFT | Acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pre-change `0d87dd9ae` | 96.51 | 9.252 ms | 108.09 | 293.3 ms | 2.877 |
| Current `6490e584f` | **98.03** | 9.252 ms | 108.08 | **252.0 ms** | 2.895 |

The mean decode difference is less than 0.01%; normal serving has no measurable
regression from route capture. The TTFT difference is favorable but is not
attributed to the source change because the jobs used different nodes.

The placement A/B uses current source. The exact `44 / 0 / 31` candidate uses
2,816 slots/rank (`44 * 64`) and deliberately leaves 54 of the 2,870 available
slots unused so that no routed layer is mixed.

| Placement | End-to-end tok/s | Mean TPOT | Decode tok/s | Mean TTFT | Acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-expert frequency `0 / 75 / 0` | **98.03** | **9.252 ms** | **108.08** | 252.0 ms | 2.895 |
| Layer-concentrated `44 / 1 / 30` | 89.75 | 10.213 ms | 97.92 | 248.1 ms | 2.865 |
| Layer-only `44 / 0 / 31` | 89.05 | 10.430 ms | 95.88 | **215.0 ms** | 2.917 |

The requested `44 / 0 / 31` layout is 2.1% slower in steady decode than
`44 / 1 / 30` and 11.3% slower than per-expert placement. Removing the last
mixed layer does not recover the cost of dropping 54 hot experts: held-out
cold-critical routes rise from 347.9 to 357.7 per token. The per-expert layout
remains the default.

Going forward, short realistic prompts are the primary performance regression
gate. Exact-400K synthetic requests remain useful for memory, DCP, and
long-context correctness stress, but not as the headline throughput baseline.

Validation: all four benchmark variants completed 24/24 requests, produced the
same semantic check (`" Paris. Distance from Paris to Lyon is"`), and captured
q4 tensors with shape `(steps, 4, 78, 8)`. Focused tests passed: scheduler route
capture 1/1; q4 grouping and hybrid budget behavior 2/2. The opt-in rejected
route capture is source commit `6490e584f` on `tiered-moe-grace-mtp`.

Jobs: capture `977597`; old profile `977598`; tail-aware `977620`; frequency
`977624`; layer-concentrated `977625`; warmed source/placement qualification
`977752`-`977755`. The jobs reuse the precompiled FlashInfer cache; cloning it
to a new absolute path invalidates Ninja metadata and triggers a redundant
180-object CUDA rebuild during startup.

Status: complete.
