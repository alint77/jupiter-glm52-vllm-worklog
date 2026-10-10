"""Phase times of one td_v30 TD_PROBE call (the last, kdev once --trace): us
from the first CTA's entry, median / max over CTAs."""
import statistics as st
import sys

import torch

r = torch.load(sys.argv[1]).tolist()
recs = [(w0 & 0xFF, (w0 >> 8) & 0xFFFF, w0 >> 24, a, b, c) for w0, a, b, c in r]
t0 = sorted(x[3] for x in recs if x[0] == 1)[-132]
recs = [x for x in recs if x[3] >= t0]
us = lambda v: (v - t0) / 1e3  # noqa: E731
f = lambda xs: f"{st.median(xs):6.1f} / {max(xs):6.1f}" if xs else "-"  # noqa: E731
k = lambda n: [x for x in recs if x[0] == n]  # noqa: E731
print(f"past PDL wait {f([us(x[4]) for x in k(1)])} | prep published {f([us(x[5]) for x in k(1) if x[5]])}"
      f" | prep seen {f([us(x[3]) for x in k(2)])} | producer end {f([us(x[5]) for x in k(2)])}"
      f" | consumer first full {f([us(x[3]) for x in k(3)])} end {f([us(x[4]) for x in k(3)])}"
      f" | work done {f([us(x[3]) for x in k(4)])} all seen {f([us(x[4]) for x in k(4)])} end {f([us(x[5]) for x in k(4)])}")
