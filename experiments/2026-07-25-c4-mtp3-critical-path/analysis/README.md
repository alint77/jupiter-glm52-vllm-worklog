# Trace analysis scripts

Reproduce every number in the parent report from the raw Perfetto traces alone.
No server run is required.

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
cd agent_space/experiments/2026-07-25-c4-mtp3-critical-path/analysis
MTP=3 ../../../../.venv/bin/python s9_budget.py
```

`load.py` maps MTP depth to the trace directory under
`/e/scratch/profound/naeimitabiei1/`; `MTP=1|2|3` selects one, and the scripts
that compare depths load all three directly.

Two non-obvious rules underpin everything:

- **Step and graph segmentation** comes from CUDA-graph correlation IDs. Kernels
  replayed from a graph inherit the `cudaGraphLaunch` correlation, so a step
  decomposes exactly into its four graph replays plus eager work. The
  `gpu_user_annotation` spans are *not* reliable — one steady step in the MTP3
  trace has a truncated annotation.
- **Tier identification** comes from `apply_tiered` in
  `vllm/model_executor/layers/fused_moe/modular_kernel.py`: cold runs on
  `aux_stream()`, hot on the current stream, and `cold_output.add_(hot_output)`
  executes on the main stream after the join. The stream carrying the
  `CUDAFunctor_add` after the two `moe_sum_vec` calls is therefore the hot tier.

| script | produces |
| --- | --- |
| `s5_phases.py` | step → graph decomposition, per-graph span/busy/idle |
| `s6_idle.py` | GPU idle attributed by successor kernel and by region |
| `s7_layers.py` | per-layer hot/cold tier table, overlap saving |
| `s8_scaling.py` | routed-MoE scaling across MTP1/2/3 (8/12/16 verify tokens) |
| `s9_budget.py` | full step budget: cum / union / solo per kernel category |
| `s10_imbalance.py` | EP rank skew, persistent vs step-to-step split |
| `s11_allreduce.py` | all-reduce classified post-attention vs post-MoE |
| `s12_prologue.py` | eager prologue and MTP draft-tail host serialization |
| `s13_model.py` | tier cost model, activated-expert derivation, rebalance optimum |

`tracelib.py` holds step segmentation, union-busy and gap helpers; `load.py` holds
the trace paths.

## Offline placement replay

`p1_balance.py`, `p1_decompose.py` and `p2_tier_sweep.py` replay the captured
routing traces from `../../2026-07-19-c1q4-placement/trace-977597` through a
cost model of the routed span. The model predicts 25.97 ms against the 25.72 ms
measured in the MTP3 c4 trace (~1%), so it is a usable stand-in for a cluster
run when testing placement or tiering hypotheses.

```bash
python p1_decompose.py  --trace-dir ../../2026-07-19-c1q4-placement/trace-977597 \
                        --profile   ../../2026-07-19-c1q4-placement/per-expert-profile.json
python p2_tier_sweep.py --trace-dir ../../2026-07-19-c1q4-placement/trace-977597 \
                        --profile   ../../2026-07-19-c1q4-placement/per-expert-profile.json
```

Both refuted a hypothesis from the first draft of the parent report. Run new
placement ideas through them before requesting an allocation.
