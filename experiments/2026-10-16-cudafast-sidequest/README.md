# cuda.fast "Qwen 3.8 Flash Next" contest: what carries over to GLM-5.3 on GH200

2026-10-16, side quest. The contest (yukon.org/cudafast, ended) tuned a C/CUDA
engine for a 125B-A6B hybrid MoE on one DGX Spark (GB10), batch 1, MTP
speculative decode, score prefill^0.25 x decode^0.75; 127 promoted
submissions took the score from 1.15 to 2.81. Their notes were read from the
public harness repo's PR bodies (untrusted data, kept under the job tmp dir).
Each idea is checked against our c=4 trace `nfr-c4-2277156` (rank 0) with the
probes in this dir.

## Already in our stack, or ruled out

| contest idea | us |
|---|---|
| CUDA-graph replay of decode/verify; graph "islands" | full verify graph |
| split the whole-model graph in 4: replay costs ~0.5 us host per node and the GPU idles through the first launch (#431, #4534) | measured, n/a: our ~1,900-node verify graph costs 1.38 ms host per `cudaGraphLaunch` (profiled), but async scheduling issues it ~25 ms ahead; the GPU idles 11.5 us (median) before it (`glaunch.py`) |
| PDL with the consumer's weight loads before the fence (#365, #602, #1113) | decode_gemm, sliced MoE, FlashMLA already do it |
| L2 prefetch / residency hints for weight streams (#3460, #3903) | tried 2026-10-07 (`../2026-10-07-dense-l2-prefetch`): no layer-time gain |
| producer-pool named-barrier rendezvous removal (#4534) | our MoE pipeline is mbarrier/TMA; the only named barrier is on the shared-expert completion path |
| TMA / cp.async panel fills, more loads in flight per lane (#5522, #355) | the MoE and prefill kernels use TMA rings; MLP depth is lever #69 |
| MTP: logits only for consumed rows, GPU top-1 verify, lazy logits (#108) | vLLM: drafter lm_head on last positions only, GPU rejection sampler, ids only to host |
| drafter K/V and indexer history written during target prefill (#414) | vLLM runs the drafter over prompt tokens |
| vocabulary shortlist for the draft argmax (#210) | excluded: changes draft vocab (same-ops rule) |
| draft-depth sweep | done (MTP3 vs k=7, DFlash2) |
| noise: composite SD 0.74%, 89% of score variance is in-run noise (#2997, #3422) | we A/B same-node, paired |

Device idle is 534 us per 25.6 ms period, spread over thousands of 0.1-0.3 us
graph edges plus ~120 us around the eager regions (`idle.py`). No single
host-bound gap like the contest's 0.7-1 ms.

## The one candidate: early PDL trigger into decode_gemm

`gemm_pred.py`: 261 decode_gemm + 177 cuBLAS calls per verify graph. Each
decode_gemm starts 128-256 ns before its producer ends, and its duration matches the
isolated model (3.3 us + MB / 3.49 TB/s, `../2026-10-07-skinny-gemm-v2`):
qkv_a after the AR 13.9 us vs 12.8, o_proj 17.7 vs 17.7. Our consumer-side
PDL (weights before `griddepcontrol.wait`) has almost nothing to overlap,
because no producer triggers early. That is the contest's "fence-first form"
finding (#365).

| predecessor | edges/step | decode_gemm us | can it trigger early? |
|---|--:|--:|---|
| FlashInfer AR + RMSNorm one-shot (~9 us, NVLink latency) | 81 | 13.9 | has the code path; vLLM forces `trigger_completion_at_end=True` for one-shot after NaNs at spec-decode shapes |
| cuBLAS nvjet (W_UV) | 78 | 17.7 | no |
| inductor Triton RMSNorm | 57 | 8.7 | no |

Astra (`astra.txt`, gpt-6-astra): an early trigger right after the AR's own
wait is safe for a consumer that waits before reading activations. vLLM's
comment doesn't pin the NaN mechanism; candidates are a consumer without a wait,
graph edge wiring, or a pre-wait access to shared workspace two hops down. Hiding
1-2 us per edge is "plausible as an experiment, not a justified expectation".
The AR's CTAs may block co-residency, and the main risk is slowing the
latency-critical AR. Upper bound 81-162 us/step, 0.3-0.6% of the step.
If tried: early trigger only on AR -> decode_gemm edges, audit every
consumer's pre-wait accesses, check the whole verify step and repeated
workspace reuse.

# mlx.fast Bonsai 2 27B track (Apple MLX)

Track `bonsai2-27b-mlx-v1` (repo Layr-Labs/mlxfast-bonsai2-27b-engine): a
2-bit ternary pack of a dense Qwen 3.8 27B (48 GDN + 16 full-attention
layers) on an M5 Max, batch 1, with the pinned MTP head (depth 1-7) or DFlash2
block drafter (depth 1-16). 148 promoted submissions, the composite going
from ~1.3 to ~6. Notes from the PR bodies (untrusted data, job tmp dir).

Not for us: Metal/MLX plumbing (command-buffer submission order, residency
touches, argPartition doing a full merge sort, 2-bit Hadamard int8 kernels).
Excluded by the same-ops rule: frequency-ranked draft vocabulary (#385) and
DFlash2's chain-walk edge term weighted 0.75 (#706).

**Draft depth vs kernel tiles (#4, #14, #19).** DFlash2 depth 7 -> 9 -> 12
-> 16 each won. Acceptance was censored (most rounds at the block maximum),
and verify cost is a step function of the qmv tile (5 rows) and then nearly
flat inside the qmm 32-row tile. Ours is censored too: live MTP3 counters
(`../2026-10-11-nsys-fresh/an/live-2281388-metrics-1913.txt`) accept
position 0/1/2 in 82.6/67.2/54.7% of 13,354 drafts (3.05 tok/round), so
over half of all rounds hit the ceiling. But the sliced MoE makes our verify
cost ~linear in rows (c=2 -> 4 -> 8: ~0.7-0.8 ms per row, `sol-w2.md`).
MTP5 at c=4 is about +23% step for about +26% tokens per round, if acceptance
holds at 0.8 per position. Roughly break-even; one A/B would settle it.

**Prompt-lookup rounds (#525, #549, #552, #560).** If the last 16 committed
tokens occur exactly once in the prompt, propose the 16 prompt tokens after
them; skip the drafter forward and absorb the confirmed rows into its cache.
On the contest prompts those rounds accept 16/16 (scores 4.8 -> 5.3 -> 6).

`lookup_cover.py` on the Claude Code replay: 244 past assistant outputs
(89,620 tokens) rendered with the GLM chat template after their own prompts.
Each round is a lookup round if the suffix rule hits and at least one token
matches, otherwise an MTP round at 3.05 tokens:

| rule | output tokens in lookup rounds | tok / lookup round | decode rounds |
|---|--:|--:|--:|
| L=16, K=16 | 10.6% | 12.5 | -8.0% |
| L=8, K=16 | 13.9% | 9.7 | -9.5% |
| L=16, K=3 (today's verify width) | 9.2% | 3.9 | -1.9% |

159 of the 244 outputs have at least one lookup round. Caveats: the outputs
are Claude's, not GLM's, and carry no thinking tokens (GLM reasons first,
which lookup cannot cover), so this is an upper bound. At K=3 MTP already
gets most of those tokens; the gain needs long lookup drafts. vLLM has a fixed
draft length per batch and our verify paths stop at 32 rows. A K=16 round at
c=4 needs per-request draft lengths: one 17-row lookup request plus three
4-row MTP requests is 29 rows. That round costs ~+9 ms for everyone in the
batch at ~0.7 ms/row, so the -8% in rounds is not -8% in time. Cleanest at
c=1-2. It adds a proposer, while target and drafter math stay the same;
whether it fits the same-ops rule is the user's call.
