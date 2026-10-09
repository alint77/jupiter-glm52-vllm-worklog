# Expert coupling (Zyphra) vs our placement (2026-10-09)

Zyphra, "Expert coupling in MoE pretraining": routing is correlated within a
layer (co-selected pairs) and across layers (layer-l expert predicts l+1);
they co-locate coupled experts per GPU to cut all-to-all bytes in training.

Our decode has no all-to-all (TP4: every GPU sees every token, owns whole
experts, all-reduce after the MoE), so co-location saves nothing; placing for
per-rank balance from calibration data was ruled out before. What could
apply: predicting cold experts, and how the hot set is ranked. Live Claude
Code captures (76 requests, 41K 8-token steps), even files learn, odd files
evaluate; prod hot set at 3,381/GPU.

## `coupling.py`

- Within a layer: the top 0.8% of pairs appear in 54-86% of tokens vs 29-71%
  under independent routing with the same loads. Real but far weaker than
  Zyphra's top-2 model (42% vs 1.6%).
- Across layers: P(likeliest l+1 expert | an l expert) median 3.5% (Zyphra:
  48%). Predicting layer l+1's cold experts at layer l, budget B experts per
  layer over the 4 GPUs, recall of the cold experts actually touched:

| | B=1 | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|--:|
| M=8 (6.3 cold/layer): coupling | 5.6% | 10.1% | 17.4% | 28.9% | 45.9% |
| M=8: frequency only | 2.9% | 5.9% | 11.0% | 20.5% | 36.1% |
| M=32 (21.4 cold/layer): coupling | 3.1% | 6.1% | 11.3% | 20.7% | 36.5% |
| M=32: frequency only | 2.6% | 5.1% | 9.7% | 18.4% | 33.5% |

  Previous step's cold set (same request): 36% / 51% recall at 6.3 / 21.4
  experts per layer. Precision ~25-35% at best: a cold prefetch would move
  3-4x the bytes it saves over C2C, which also carries demand cold reads and
  the Grace KV. Not worth building.

## `hotrank.py`: rank the hot set by steps touched, not routes

A cold expert costs one load per step whatever the number of tokens routed to
it; the 8 verify tokens of a request are consecutive and route alike. Same
owners and hot count per GPU, held-out cold experts per GPU-layer (mean /
slowest GPU of the layer):

| hot set ranked by | M=8 | M=32 |
|---|--:|--:|
| prod profile (MiMo-task lists + CC route-count promotion) | 1.549 / 2.780 | 5.306 / 7.393 |
| route count, per GPU over layers (prod's promotion rule) | 1.328 / 2.464 | 4.723 / 6.820 |
| steps touched at M=8, per GPU over layers | **1.182 / 2.264** | **4.279 / 6.284** |
| steps touched at M=32, per GPU over layers | 1.185 / 2.270 | 4.290 / 6.283 |

-11% / -9% cold vs route counts on the same data, slowest GPU -8%. At
~17 us per cold-instead-of-hot expert (`../2026-10-08-m32` fit): ~-0.25 ms/step
at 8 tokens, ~-0.7 at 32, before the replay-to-served shrink seen last time
(`../2026-10-08-moe-cost-table`: -0.9 replayed, -0.2 served).
