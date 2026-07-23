# C4 MTP-depth and residency-edge sweep

Goal: compare MTP1, MTP2, and MTP3 at concurrency four on the qualified
TP4/EP4/DCP4 V2-runner path, while minimizing Grace expert offload.

All three configurations start at the tiered runtime's minimum legal HBM
reserve: 7 decimal GB/rank. Relative to the prior 10 GB c4 runs, this makes
about 3 GB/rank more HBM available to routed experts. The domain-trained
per-expert profile remains the ordering seed; the planner deterministically
promotes additional cold experts into the newly available capacity.

Each job runs:

1. semantic and exact-400K SHA correctness gates;
2. two cache-cold repetitions of the 24-prompt Python/PyTorch/ML/math suite at
   concurrency four;
3. two matched 4K-input, 256-output c4 repetitions;
4. a prefix-primed 395,904-input c4 decode measurement; and
5. an eight-step, four-rank PyTorch profile of that full-context c4 decode.

Raw profiler traces are written to scratch and their paths are recorded with
the final results.
