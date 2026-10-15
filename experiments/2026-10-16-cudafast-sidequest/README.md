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
