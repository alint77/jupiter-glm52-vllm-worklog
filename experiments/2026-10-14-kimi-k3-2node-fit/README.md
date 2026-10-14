# Kimi K3 on 2 Booster nodes: does it fit?

Question (2026-10-14): can moonshotai/Kimi-K3 (sha f831ab66) be served on 2
nodes (8 GH200), e.g. EP4-TP4-DCP4-PP2? Paper sizing only, nothing run.

## Weights (exact, from every shard's safetensors header: `hdr.sh`, `k3sum.py`, `sizes.txt`)

| class | GiB |
|---|---:|
| routed experts, MXFP4 (92 MoE layers x 896, 16.73 MiB each) | 1347.1 |
| attention bf16 (69 KDA ~0.83 GiB/layer, 24 MLA ~0.43) | 67.4 |
| shared experts bf16 | 22.6 |
| latent up/down proj, routers, norms bf16 | 9.9 |
| embed + lm_head | 4.4 |
| dense layer 0 MLP | 1.4 |
| vision tower | 0.8 |
| **total** | **1453.7** |

## Memory on 2 nodes (measured, not datasheet)

- HBM: 95.0 GiB per GPU. GLM prod leaves ~85 GiB per GPU for weights after
  activations, graphs, non-torch and the observed reserve (server-claude-2281388).
  8 GPUs: ~680 GiB.
- Grace: ~119 GiB per superchip, but an idle node already has 18-42 GiB per Grace
  used (374 GiB free per node, 2026-09-26-mimo-decode-profile/grace-free-2074487),
  and GLM's loader sees ~315 GiB available with vLLM up. 2 nodes: 630-750 GiB.
- Usable total: ~1310-1430 GiB < 1454 GiB of weights alone, before KV, KDA
  state, prefill staging slots or any slack.

The parallel layout cannot change the total. EP4/TP4/DCP4 inside a node and PP2
across is the right shape (only the 7168-wide hidden state crosses IB), but it
does not create memory.

## KV is not the problem

24 MLA layers x 656 B (fp8) = 15.4 KiB per token for the whole model; KDA state
~434 MiB per request (fp32, all 69 layers). With PP2 and DCP4, 1M tokens is ~1.8
GiB per GPU.

## 3 nodes fit

PP3 x (TP4/EP4, DCP4): ~2000 GiB usable for 1454. Per node 449 GiB of experts:
~260 GiB hot in HBM (~65 GiB/GPU), ~190 GiB cold in Grace (vs ~315 available).

## Software

Upstream main (5f30fc7031) supports K3 (SupportsPP, a Hopper KDA path, MXFP4
latent MoE, EAGLE3/MTP drafters). It has no Grace tiering. Our tiered MoE is
INT4/Marlin; K3 needs MXFP4 (e2m1, e8m0 scales, g32) decode/prefill kernels, the
latent MoE (experts on 3584 dims between bf16 up/down projections), 896 experts
top-16, and multi-node PP.
