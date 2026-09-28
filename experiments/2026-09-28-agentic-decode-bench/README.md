# GLM-5.3 vs MiMo-V2.6 decode on the MiMo agentic task set (2026-09-28)

Every earlier GLM number (the `ab-*` campaign in `../2026-09-27-glm53-mtp7-profile`)
was measured with `bench.py`: greedy, `/v1/completions`, raw prompts, `ignore_eos`.
That text loops -- MTP accepted 6.5-7 of 8 drafts, and the hot set covered 74% of
routes against 90% on real traffic -- so its absolute numbers (36 ms, "170 tok/s")
and the "GLM MoE is 2x MiMo's" conclusion do not describe real serving. MiMo's
18.2 ms came from yet another harness (chat endpoint, fixed prompts).

## Harness

`agentic_bench.py`: the MiMo capture task set
(`../2026-09-26-mimo-routing-profile/tasks-{0..3}.json`, 16 tasks, 22 turns) through
the capture driver's system prompt, tools and agent loop
(`../2026-08-29-glm53-routing-capture/run_agentic_capture.py`), on
`/v1/chat/completions`, temperature 1.0 / top_p 0.95 (both models' defaults), a
seed per request. Per request it diffs the server's counters: decode time, verify
steps, accepted tokens. `bench_node.sh glm|mimo <tag>` serves the production config
of either model (GLM: `serve.sh` DCP4 MTP7 + agent parsers; MiMo:
`serve.sbatch` + `claude-server.sbatch`'s args). `compare.py` fits
`step_ms ~ a + b*tokens/step + c*ctx` and drops requests that hit the 8192-token
cap (looping). With `--profile N` and `TRACE_ROOT`, it takes N torch-profiler
windows mid-decode (`launch_prof.sh`), each in its own `window-<i>` dir.

Findings on the way:
- **GLM DCP4 + prefix caching crashes** on the first prefix hit:
  `cp_gather_cache ... src_cache and dst must have the same dtype` (MLA
  `_context_parallel_compute_prefill_context`). GLM runs here with prefix caching
  off (does not affect decode steps). Must be fixed before DCP4 can serve Claude
  Code.
- MiMo sometimes loops to the output cap (3 of 46 requests, 7.4-7.6 tokens/step);
  GLM did not (0 of 175).
- The harness's `grep` tool scans the whole checkout (incl. `.venv`), up to its
  60 s timeout per call; MiMo's run spent most of its wall time there.

## Step time (rows-glm-tasks2, rows-mimo-tasks{1,2})

    GLM  96 requests (>= 20 steps), 11,754 steps, ctx median 5.2K:
         27.87 ms/step at 3.41 tokens/step -> 122 tok/s decode
         fit 27.02 + 0.09/token + 0.44/10K ctx -> 27.70 ms at 3.5 tok, 8K
    MiMo 43 requests, 7,503 steps, ctx median 8.6K:
         17.50 ms/step at 3.53 tokens/step -> 201 tok/s decode
         fit 14.43 + 0.58/token + 0.53/10K ctx -> 16.90 ms at 3.5 tok, 8K

GLM accepts 3.4-3.5 tokens/step (the user's Claude Code experience: 2-3), and its
step time is flat in acceptance on real text (+0.09 ms/token): the greedy bench's
+1.8 ms/token was its looping text.

## Where the 9-11 ms goes (profiles: traces/glm53-agentic-2102013, mimo26-agentic-2102833)

`analyze.py` per window, ms/step, first two windows each (73/71 GLM, 58/46 MiMo
steps). PDL is on (production), so per-kernel MoE spans overlap; only the MoE
chain's total is meaningful. `triton_tem_fused_mm_*` (Inductor matmul templates
with a fused norm, 0.74 ms on GLM) are GEMMs that `analyze.py` files under
norm/rope/elementwise; moved here.

| | GLM | MiMo | diff |
|---|---:|---:|---:|
| MoE: one-kernel chain + routing | 8.2 | 8.8 | -0.6 |
| drafter (MTP7: 7 sequential passes; DFlash: 1) | 3.6 | 0.9 | +2.8 |
| dense GEMM (incl. Inductor mm templates) | 5.0 | 3.0 | +2.0 |
| small unfused glue kernels | 2.3 | 0.6 | +1.8 |
| DCP one-shot collectives | 1.7 | 0 | +1.7 |
| attention + DSA indexer | 2.5 | 1.1 | +1.4 |
| TP all-reduce | 2.2 | 2.3 | 0 |
| idle + host-side + logits | 2.1 | 1.9 | +0.2 |
| step period | 27.6 | 18.4 | +9.2 |

GLM's glue, per step: 252 `FillFunctor<int>` zero-fills, 153 `elementwise_kernel`,
78 `CatArrayBatchedCopy`, 78 int->bool casts, 78 `_correct_attn_cp_out_kernel`
(DCP's attention merge), shared-expert silu, ... MiMo's: 3 fused norm kernels.
