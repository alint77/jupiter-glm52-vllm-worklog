**The fix addresses the reported race.** The supplied SASS and old/fixed test results strongly support it. In priority order:

1. **Correctness: ordinary `ld.global` after the wait is sufficient.** `griddepcontrol.wait` guarantees prerequisite-grid completion and visibility of their memory operations. That covers ordinary `.ca` loads: stale L1 contents cannot legally defeat that guarantee. You need neither `.cg` nor `.relaxed.gpu`; `.cg` controls caching, and `.relaxed` does not replace synchronization. This conclusion follows from the specified visibility guarantee, without assuming a particular hardware invalidation mechanism. `ld.global.nc` is explicitly outside the PTX memory-consistency model. [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)

2. **Medium: strengthen regression coverage.** The old build’s failures demonstrate that the producer actually exposes the hazard. Comparing against the same configuration with `pdl=False` is appropriate: arithmetic and reduction order are unchanged. Add:
   - **`NT=3`**, currently missing; use `M=17,24`, plus partial-tile cases such as `M=9,25`.
   - **Alternating source values across invocations.** Identical `src` values can hide cached data from a previous successful invocation despite the sentinel stores.
   - A producer that writes one region early and another late, with differing CTA delays; also queue multiple producer/consumer pairs without per-pair host synchronization.

   Four producer CTAs suffice for this reproduction. For *literal* bitwise equality, compare `int16` views; floating `torch.equal` treats signed zeros as equal.

3. **Low: make the inline-asm memory contract explicit.** NVIDIA documents `asm volatile` as preventing deletion/movement during PTX generation, supporting this sequence of volatile statements. Nevertheless, `ldx` hides a memory read; I would add the clobber:
   ```cuda
   : "l"(p) : "memory");
   ```
   This is hardening, not evidence that your fixed binary remains broken. At the PTX-to-SASS stage, moving ordinary dependent loads across the wait in a way that changes results would violate its semantics. Keep the SASS check across compiler upgrades. Also change the comment to “volatile inline asm issuing ordinary global loads”: these are not PTX `ld.volatile` loads. [NVIDIA inline-PTX guidance](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#incorrect-optimization)

4. **Weights and early triggers: agree with the pattern.** Immutable, already-initialized weights can load before the wait. Triggering dependents at producer entry—and before this kernel’s output epilogue—is valid: the trigger permits scheduling; downstream waits establish readiness. Every thread consuming predecessor-produced data must be properly synchronized. No extra producer fence is needed for this wait-based pattern. [CUDA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

   For the tiered-prefill exception, disjoint rows alone are insufficient: the pre-wait list and every nonwaiting CTA’s inputs must already be ready independently of the overlapping predecessor.

5. **Performance: measurable differences are plausible.** Vector width stays 16 bytes, but the small unique footprint understates traffic. Across CTAs, activation-load bytes relative to weight-load bytes are approximately **`M/16`**, reaching **2× at `M=32`**, before cache reuse. Cache behavior, scheduling, and register pressure can therefore matter despite L2 residency. Benchmark the final helper, including any clobber, and include `pdl=False`, which this patch also changes.

6. **Separate, pre-existing robustness issue:** if callers do not enforce them, validate contiguous CUDA bf16 inputs, matching dimensions/devices, 16-byte alignment, and positive `K`. In particular, **`K=0` passes the divisibility check but still executes the initial loads**. This does not affect the supplied production shapes.
