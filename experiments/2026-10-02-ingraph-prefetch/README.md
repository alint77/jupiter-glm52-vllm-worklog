# In-graph cold prefetch (GLM-5.3, prefill graphs to 1024)

Follows ../2026-09-30-glm-prefill-prefetch (plan section there).

## Capture crash (e3a4143956) -- root cause

`server-*.out` (not `.err`) had the worker traceback:
`cudaErrorStreamCaptureIsolation` in `join_captured`. Piecewise capture makes
one graph per layer; the last MoE layer forks no copy but still joined the copy
stream, which was outside its capture. Fixed in daeecd8bdf (join only after a
fork); the capture test now captures one graph per layer and fails without it.

## Same-node A/B (jpbo-001-44, job 2138596), median TTFT ms +- std, 20 prompts

| arm | 512 | 768 | 1024 |
|---|---|---|---|
| nopf (threshold 1025) | 176 +-3 | 236 +-12 | 240 +-12 |
| inmoe (daeecd8: fork at MoE(L), join after it) | 143 +-1 | 174 +-12 | 203 +-10 |
| wide (fork before o_proj(L), join before attn(L+1), first MoE layer under the dense layer) | 135 +-28 | 172 +-7 | 205 +-13 |

- Prefetch in graphs: -33 to -64 ms. Widening the window: within noise; the
  copy is already mostly hidden under MoE(L) at >=512 tokens.
- Greedy text is not a correctness check here: identical no-prefetch configs
  on two nodes (agentic-bench/greedy-gp-nopf-n{1,2}) already differ.
  Correctness: VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY now also runs each
  captured MoE against Grace and keeps max |diff| on the device, reported by
  the next eager chunk (launch_verify.sh).

## Trace: whole-piece design (trace.sh wide, analyze_trace.py), 4 ranks

One traced prefill each at 512/768/1024 new tokens. Graph replays put every
node on the launch stream; copies are the >=5 MB HtoD at ~400 GB/s.

| tokens | copy L+1 (us, median) | MoE(L) (us, median) | layers where copy > MoE(L) | join stalls (layers, total) |
|---|---|---|---|---|
| 512 | 1350-1400 (~510 MB) | 950-1080 | 53-62 / 73 | 10-17, 1.1-2.7 ms |
| 768 | same | 1160-1310 | 34-41 / 73 | 0-2, <20 us |
| 1024 | same | 1430-1600 | 9-27 / 73 | 0 |

- The MoE-only window is too short for most layers at 512, half at 768, and a
  minority at 1024; the wider window hides all of it from 768 up, and at 512
  still stalls 1-3 ms per prefill (forward ~124 ms first-to-last MoE).
- Dense-layer copy (first MoE layer): 320-440 us, no stall.
