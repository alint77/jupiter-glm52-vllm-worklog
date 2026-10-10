"""td_v41.cu = td_v40.cu with the producer's per-tier sizes kept in registers:
`len[q]` / `n_tier[q]` indexed by the runtime tier q were a 16 B stack frame,
two LDLs per group, and the w2 ready spin's ld.acquire.gpu emits CCTL.IVALL
(L1 invalidate), so those LDLs went to L2 (~1 us per w2 unit at M=8).
TD_NO_RECPF still turns v40's record prefetch off."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v40.cu").read_text()
n = src.count("len[q]") + src.count("n_tier[q]")
assert n >= 4, n
src = src.replace("len[q]", "(q ? len[1] : len[0])")
src = src.replace("n_tier[q]", "(q ? n_tier[1] : n_tier[0])")
(here / "td_v41.cu").write_text(src)
print("ok", n)
