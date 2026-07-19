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

## Results

All runs used c1, MTP3/q4, 2,870 HBM expert slots per EP rank, a 10 GiB
HBM reserve, 24 prompts, and 256 forced output tokens per prompt.

| Placement | Full / mixed / cold layers | Output tok/s | Mean TPOT | Acceptance length |
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

Validation: all four benchmark variants completed 24/24 requests, produced the
same semantic check (`" Paris. Distance from Paris to Lyon is"`), and captured
q4 tensors with shape `(steps, 4, 78, 8)`. Focused tests passed: scheduler route
capture 1/1; q4 grouping and hybrid budget behavior 2/2. The opt-in rejected
route capture is source commit `6490e584f` on `tiered-moe-grace-mtp`.

Jobs: capture `977597`; old profile `977598`; tail-aware `977620`; frequency
`977624`; layer-concentrated `977625`. The jobs reuse the precompiled FlashInfer
cache; isolated empty caches had otherwise triggered a redundant 180-object
CUDA rebuild during startup.

Status: complete.
