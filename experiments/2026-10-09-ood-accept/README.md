# MTP3 vs DFlash2 acceptance off the coding distribution (2026-10-09)

Question: is DFlash2 overfit to agentic / coding text? 16 prompts
(`prompts.json`: 8 code - PyTorch autograd, decode roofline, paged KV cache,
AVX-512 C++, CUDA transpose, Rust SPSC, SQL windows, React; 3 maths; 3 prose;
French; medical), 3 seeds x 512 tokens, temperature 1.0 / top_p 0.95, one
request at a time (`ood_accept.py`, server spec-decode counters per request,
totals and per position). Two modes: `content` = the chat prompt rendered by
the model's template with the think block closed, via /v1/completions;
`thinking` = /v1/chat/completions as served. c=1 servers on prod serve.sh
(`ood_arm.sh`, `launch.sh`): MTP3 (reserve 3.6), DFlash2 k=3, DFlash2 k=7.
`analyze.py` prints everything; `plot_ood.py` makes the write-up figure.

Answer mode, tokens per step (of 4 at k=3):

| domain | MTP3 | DFlash2 k=3 | gap | DFlash2 k=7 |
|---|--:|--:|--:|--:|
| code (8) | 2.63 | 2.53 | -0.10 (-4%) | 3.15 |
| maths (3) | 2.90 | 2.74 | -0.16 (-6%) | 3.76 |
| prose (3) | 2.01 | 1.82 | -0.18 (-9%) | 1.95 |
| other (2) | 2.68 | 2.39 | -0.29 (-11%) | 2.85 |
| all | 2.53 | 2.37 | -0.16 | 2.87 |

Thinking mode: all 2.55 / 2.42 / 2.89; maths 2.68 / 2.78 (DFlash2 ahead).

Per position (answer mode), accepted at i / drafts: MTP3 0.72 / 0.49 / 0.32,
DFlash2 k=3 0.63 / 0.44 / 0.31; conditional on i-1 accepted: MTP3 0.72 /
0.68 / 0.65, DFlash2 0.63 / 0.70 / 0.70. The deficit is the first draft token.
Per-request spread over seeds 0.2-0.36 tokens/step: single-prompt gaps are
noise, the domain gradient and the overall -0.16 (48 requests each) are not.
