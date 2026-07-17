# Tiered Marlin decode stream overlap

Four-rank batch-one run on JUPITER allocation `958529`. This changes only the
hot/cold routed-expert schedule from serial to overlapped execution; model,
placement, 400K HBM MLA cache, compile configuration, and benchmark shape match
the preceding compiled HBM-cache run.

## Minimal implementation

The two Marlin calls receive four non-overlapping views from one existing vLLM
workspace allocation: one reusable common/output view and one activation view
per tier. Cold Marlin runs on vLLM's existing auxiliary CUDA stream while hot
Marlin runs on the current stream. The current stream waits once, adds the two
partials in place, and uses the existing finalize path.

Overlap is limited to the captured one- and two-token decode shapes. The first
attempt also doubled the 8,192-token prefill workspaces; attention warmup then
failed while requesting 896 MiB with only about 508 MiB free. Keeping large
prefill sequential removed that extra peak without adding an allocator or
configuration surface. The final startup retained 4.28 GiB free HBM per GPU.

## Validation

- Ruff check and format passed.
- Python 3.12 mypy passed.
- The focused tiered suite passed 29/29.
- Piecewise graphs for sizes one and two and the full decode graph captured.
- Two deterministic 5-input/8-output requests produced the exact prior text:
  ` Paris. Distance from Paris to Lyon is`.

## Results

Each run uses one random 4,096-token prompt, 256 forced output tokens, and
concurrency one. Decode rate is the reciprocal of mean TPOT.

| Schedule | TTFT | TPOT | P99 ITL | Output tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial, seed 0 | 1.003 s | 29.03 ms | 29.66 ms | 30.45 | 34.45 |
| Overlap, seed 0 | 1.006 s | 27.56 ms | 28.14 ms | 31.86 | 36.28 |
| Overlap, seed 1 | 0.957 s | 27.54 ms | 28.28 ms | 32.08 | 36.31 |

The two overlap samples agree within 0.02 ms TPOT. Decode improves by about
5.4% and is now roughly 2.0% below the 37.06 tok/s native-offload baseline.
This closes the fixed serial two-tier cost without changing output. It does not
close the project's larger throughput target, which the native baseline also
does not meet.

After both 4K requests, physical free HBM was 1,500-1,503 MiB per GPU because
the allocator retained prefill buffers. The existing high-water reserve caveat
therefore remains.
