"""skinny_gemm correctness driver for compute-sanitizer runs.

Covers every config at shapes where a CTA owns whole tiles, where split-K
spreads one tile over several CTAs (the atomics + last-CTA flush path), and a
ragged N/K edge. Asserts against torch and that the zero-on-exit workspace
contract holds.
"""

import sys

import torch

from vllm.model_executor.layers.skinny_gemm import _workspace, skinny_gemm

torch.manual_seed(0)
dev = torch.device("cuda:0")
cases = [
    # (M, N, K, config, grid)
    (8, 128, 512, 1, 2),  # whole tiles per CTA
    (8, 128, 512, 1, 3),  # uneven split-K
    (5, 256, 1024, 0, 7),
    (1, 192, 768, 2, 5),
    (8, 200, 520, 3, 4),  # ragged N and K
    (8, 6144, 2048, 0, 0),  # one CTA per SM
]
bad = 0
for m, n, k, cfg, grid in cases:
    x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
    for _ in range(2):
        got = skinny_gemm(x, w, cfg, grid).float()
    want = x.float() @ w.float().t()
    err = ((got - want).abs().amax() / want.abs().amax()).item()
    ws, counters = _workspace(dev, n)
    dirty = int(ws.count_nonzero()) + int(counters.count_nonzero())
    ok = err < 1e-2 and dirty == 0
    bad += not ok
    print(f"M={m} N={n} K={k} cfg={cfg} grid={grid}: err={err:.2e} dirty={dirty}",
          "ok" if ok else "FAIL")
sys.exit(1 if bad else 0)
