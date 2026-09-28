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

`pool_windows.py`, all four windows each (1,000 GLM / 1,220 MiMo rank-steps):

| | GLM | MiMo | diff |
|---|---:|---:|---:|
| MoE: one-kernel chain + routing | 8.17 | 8.88 | -0.71 |
| drafter (MTP7: 7 sequential passes; DFlash: 1) | 3.63 | 0.87 | +2.76 |
| dense GEMM (incl. Inductor mm templates) | 5.05 | 2.97 | +2.08 |
| small unfused glue kernels | 2.31 | 0.56 | +1.75 |
| DCP one-shot collectives | 1.72 | 0 | +1.72 |
| attention + DSA indexer | 2.47 | 1.07 | +1.40 |
| TP all-reduce | 2.26 | 2.37 | -0.11 |
| GPU idle + host-side + logits | 2.16 | 1.85 | +0.31 |
| step period | 27.78 | 18.58 | +9.20 |

GLM's glue, per step: 252 `FillFunctor<int>` zero-fills, 153 `elementwise_kernel`,
78 `CatArrayBatchedCopy`, 78 int->bool casts, 78 `_correct_attn_cp_out_kernel`
(DCP's attention merge), shared-expert silu, ... MiMo's: 3 fused norm kernels.

## GLM with DCP off at MiMo's 250K context (rows-glm-dcp1-250k, job 2104973)

`launch_dcp1.sh`: DCP=1, MAX_MODEL_LEN=250000 (VLLM_TIERED_MOE_RELAX_SHAPE=1 lifts
the 400K pin), prefix caching on (the crash is DCP-only), everything else as the
DCP4 run. 178 requests, no errors; 6 ran to the output cap (DCP4: 0 -- sampling).

    DCP1 250K  122 requests, 21,591 steps: 28.95 ms/step at 3.35 tok/step, 116 tok/s
               fit 25.43 + 0.83/token + 0.62/10K ctx -> 28.83 ms at 3.5 tok, 8K
    DCP4 400K  (above) -> 27.70 ms;  MiMo -> 16.90 ms

DCP4 stays ~1.1 ms/step faster even at the shorter context. DCP1 holds the full
MLA cache per GPU, so 2,815 hot experts per GPU against DCP4's 3,208, and pads
16 local heads to 64 in the sparse FlashMLA kernel; its step time also rises
with acceptance (+0.83 ms/token, DCP4 +0.09), consistent with more cold experts
per step. Profile windows: the first profile node failed to launch its step
(Slurm), rerun as `launch_dcp1_prof.sh`.

## Prefix caching under DCP4: fixed (vLLM 9526fb671f)

Sparse MLA takes dense (MHA) prefill when the whole sequence fits in
`index_topk` (2,048), so a prefix-cache hit on a short prompt reads its cached
context through `_compute_prefill_context` / `_context_parallel_compute_prefill_context`.
Neither generic gather understands fp8_ds_mla's 656-byte entry (512 fp8 values,
four fp32 tile scales, 64 bf16 rope values). `probe_ds_mla_gather.py`, one GPU:

    (b) cp_gather_and_upconvert_fp8_kv_cache (sparse backend's reader): rel err kv_c 0.0259, k_pe 0
    (a) gather_and_maybe_dequant_cache('fp8_ds_mla') (non-DCP path):     rel err kv_c 321, k_pe NaN
    (fix) _gather_ds_mla_context, 2 requests, mid-sequence starts:     rel err kv_c 0.0258, k_pe 0

So without DCP the context was silently garbage -- this is the path the current
DCP1 production launcher runs with prefix caching on -- and with DCP it failed
the dtype check. The fix gathers the raw entries (`cp_gather_cache`, uint8) and
decodes them; regression test `tests/kernels/attention/test_cache.py::
test_gather_ds_mla_context`.

End to end (rows-glm-dcp4-prefix, job 2106177): DCP4 400K prefix caching on,
166 requests, no errors, 77.9% prefix hit rate. Decode unchanged -- 28.08 ms at
3.5 tok / 8K (prefix off: 27.70; 28.16 vs 27.87 step-weighted, within run
noise) -- and total prefill 70 s against 298 s without the cache.
