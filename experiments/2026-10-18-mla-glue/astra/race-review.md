**The strongest lead is the shortened lifetime of `qkv_dbg`, the input to glue. The supplied excerpts do not establish the offending overwrite, so I would not yet call this a proven allocator race. They do reveal that the current `NOPDL` probe is incomplete.**

Two observations materially change the diagnosis:

- Every rescue path rereads `qkv_dbg`: `mla.py:159` for QORIG, `:162` for KVORIG, and `:168–170` for KROPE. All extend the projection output’s live range and can change buffer reuse.
- KROPE replaces the q-only RoPE call with a q-and-k call. It does **not necessarily add a kernel**. Its effect could be lifetime, aliasing, allocation, or compiled scheduling.
- The table’s KVORIG row is cumulative: “**+** kv_c_normed.” It does not establish that KVORIG alone restores quality.

Locations below use the supplied pf9 numbering for `mla.py` and `deepseek_v2.py`; new-file locations use `glue.diff` numbering unless stated otherwise.

1. **Highest-ranked: fusion exposes premature reuse of the projection output’s storage.**

   The relevant transition is `mla.py:145–155`: without a debug recomputation, glue becomes the last explicit reader of `qkv_dbg`. Glue reads that allocation at `mla_glue.cu:72–74` and again for RoPE at `:114–115`.

   The precise hypothesized failure is:

   ```
   glue is still reading allocation A, containing qkv
   → another operation acquires or explicitly reuses A
   → that operation writes A without waiting for glue’s reads to finish
   → glue produces corrupted norms and/or cache entries
   ```

   Each recomputation extends A’s live range, potentially removing that reuse opportunity. This explains why replacing a downstream tensor that decode does not read can restore quality.

   **The missing piece is the unordered writer.** Releasing a tensor after enqueueing its last consumer is ordinarily safe on the same stream. Moreover, if `q_b` writes its output only after a successful dependency wait, reusing A for that output is safe. The observed pre-wait weight loads do not establish a hazard against A.

   Consequently, the excerpts identify the suspicious lifetime transition, but **do not contain an offending free/reuse/access line**. That requires the emitted compiled wrapper, the actual GEMM implementation, or the relevant side-stream implementation.

   **Cheapest discriminating probe:** extend only `qkv_dbg`’s lifetime, retaining all three glue outputs and adding no GPU computation. Use a compiler-visible keepalive through attention, verified to survive compilation and enqueue no CUDA kernel. Alternatively, retain its owning allocation through graph lifetime inside the opaque implementation, but check that Inductor does not explicitly reuse the storage despite that reference.

   Run the quality comparison on Booster. On the login GPU, inspect allocation addresses and reproduce the compiled/captured lifetime pattern with repeated replays; ordinary isolated-op tests do not exercise it.

   **Fix if confirmed:** repair the actual ownership or ordering violation. A side-stream consumer needs producer-to-consumer ordering and protection against storage reuse until it finishes. A PDL consumer must put accesses to reused storage after its wait. An explicit compiler reuse requires a correctly represented use/dependency. Permanent retention is a diagnostic workaround.

2. **Second: an output or neighboring workspace is reused across an unrepresented asynchronous use.**

   The allocation sites are `mla_glue/__init__.py:133–135`—pf9 `:140–142`. These allocations are legal in a custom op and during capture. Their being allocator-managed does not itself establish a bug.

   The hypothesized failure is either:

   - glue writes a returned buffer whose previous asynchronous user is still active; or
   - a returned buffer is reused while an asynchronous consumer still reads it.

   Changing any output’s lifetime can alter allocator placement and mask either failure. The common `qkv_dbg` lifetime change makes hypothesis 1 more specifically supported.

   **Do not treat `q_b` as the last reader of `q_c`.** The wrapper passes it to the indexer at `mla.py:223`, and `deepseek_v2.py:744` reads it in `self.wq_b(qr)`. A proposed lifetime explanation must account for that explicit use.

   **Cheapest discriminating probe:** separately retain all three glue output allocations for the graph’s lifetime, without recomputation or extra device work. Compare this against retaining only `qkv_dbg`. Dedicated per-layer, per-graph output buffers are another useful arm; they must not be shared by concurrently executing graphs.

   If output retention restores quality while input retention does not, examine the addresses and ownership of those outputs and any storage they replaced. Compare `q_c` at its two consumers using preallocated diagnostic storage, recognizing that instrumentation can perturb timing.

   **Fix if confirmed:** correct the missing stream dependency, lifetime tracking, or compiler-visible alias/mutation contract. Caller-provided outputs can make ownership easier to control, but cannot substitute for a required stream wait.

3. **Third: the fused cache writes cross a cache/stager readiness boundary.**

   This change moves cache writes from `mla_attention.py:1130–1137` into `mla_glue.cu:139–143`.

   In wrapper order, that moves them ahead of:

   - `q_b` and RoPE;
   - the cold-prefetch join at `mla.py:220`;
   - the indexer at `mla.py:223`.

   The stager buffer and map are obtained at `mla_glue/__init__.py:110–117`. Neither their readiness nor the cache mutation appears as an explicit input/mutation in the registration at `:173`.

   A concrete failure would be:

   ```
   another stream is still preparing reserved_map or staging storage
   → glue reads the old map or writes storage still owned by that operation
   → attention later reads stale, overwritten, or incorrectly placed entries
   ```

   The supplied stager and prefetch code is insufficient to establish that ordering violation. The existing cache-update dummy dependency also still connects glue outputs to attention through `mla_attention.py:637–652`; the graph is not simply missing every dependency from glue to attention.

   **Fit limitation:** a permanently wrong map or an already-corrupted cache entry is a weak explanation for KROPE. A later recomputation cannot repair an earlier wrong write merely by delaying attention. This hypothesis needs an overlapping writer/reader, changed compiled scheduling, or changed storage placement.

   **Cheapest discriminating probe:** preserve compilation and FULL graphs, but make glue compute only its three outputs and restore the original cache update, including the stager’s `write_new` path. Change both sides together so there is exactly one writer.

   This isolates the moved cache-write path much better than `--enforce-eager`. A successful result still needs a follow-up: restoring the original writer also restores its scheduling effects.

   **Fix if confirmed:** place the fused writes after the actual map/stager readiness dependency, preserve the stager ownership protocol, and represent relevant mutations/dependencies explicitly. Keeping the original cache writer is a reasonable interim implementation.

4. **Lower-ranked: capture-path mismatch, fixed cache-address error, or arithmetic/code-generation error.**

   The relevant gating is `mla_glue/__init__.py:78–87`, paired with `mla_attention.py:1110–1114`.

   A capture-time decision is baked into the captured launches. `is_current_stream_capturing()` being false during ordinary execution does **not** imply that graph replay loses its captured cache writes. Python does not rerun those decisions inside a replay.

   The predicate test nevertheless proves only the predicate’s behavior. It does not prove that every captured graph contains exactly the intended main-cache and stager writers or binds the correct persistent buffers.

   **Probe:** record the actual capture-time writer decisions and cache/map bindings for each graph/layer, then inspect replay correctness on the login GPU. Fixed missing writes or wrong bindings should remain wrong under QORIG/KROPE unless those variants alter capture selection or scheduling, making this a weaker fit.

   Arithmetic is weaker still: isolated tests are strong evidence, and a numerically unused KROPE result has no direct route to correct glue’s cache bytes. A compiled lowering bug remains possible because the tests do not establish identical generated code for all served variants.

   **Fix:** repair any demonstrated capture binding or generated-code discrepancy; there is no specific such defect established here.

The evidence distinguishes these candidates as follows:

| Evidence | Input lifetime/reuse | Output/workspace lifetime | Cache/stager ordering | Fixed math/binding error |
|---|---|---|---|---|
| Isolated op tests pass | Expected; reuse pattern absent | Expected; asynchronous ownership absent | Expected; readiness protocol absent | Weakens math; does not test served bindings |
| Prefill is clean | Compatible with different lifetimes and scheduling | Compatible with different graph/stream behavior | Compatible with FULL-only fused writes | Weakens a general arithmetic defect |
| Every full-glue server degrades | Compatible with a consistently exercised hazard | Compatible | Compatible | Compatible by itself |
| KROPE restores near-prod | Strongest clue: retains `qkv_dbg` | Can change allocation and scheduling | Needs a timing/scheduling mechanism | Poor fit for glue arithmetic or a fixed wrong write |
| QORIG restores near-prod | Retains `qkv_dbg` | Changes output lifetime and supplies both q consumers | Needs scheduling effects | Supports a q-specific error alone, but does not explain KROPE |
| Cumulative KVORIG is good | Compatible; adds no independent isolation | Compatible | Compatible | Does not establish a KV-only rescue |
| NOPDL remains degraded | **Current knob does not exclude PDL-related reuse** | Same; non-PDL stream hazards also remain | Compatible | Compatible |
| NODRAFT / both | Base-model hazards remain | Base-model hazards remain | Base-model hazards remain | Excludes an exclusively draft-layer cause |
| Larger sampling/MTP impact | Downstream amplification is plausible | Same | Same | Magnitude alone does not localize the error |

The sampled NOPDL and “both” results improve substantially; “both” is inside the reported prod sampled range. Their greedy results remain degraded. With one server each, those arms do not support a clean blanket exclusion.

**The definite diagnostic bug is at `mla_glue.cu:215–216` in pf9.** `VLLM_MLA_GLUE_NOPDL` only removes glue’s `cudaLaunchAttributeProgrammaticStreamSerialization`. That controls whether glue can launch early behind its predecessor.

Glue still executes:

```cpp
asm volatile("griddepcontrol.launch_dependents;");
```

at `mla_glue.cu:66`. A PDL-enabled `q_b` can therefore still launch early behind glue. Your NOPDL arm does not serialize both sides of glue.

Separately, signaling before the final stores is valid when dependent accesses occur after the required wait. The early signal and observed overlap are not themselves the race.

**What to do first:**

1. **Repair the serialization probe.** Disable glue’s PDL launch attribute **and** its early `launch_dependents` signal. Keeping its wait is fine. Verify that glue starts after the projection finishes and `q_b` starts after glue finishes. This changes neither tensor values nor intended allocation lifetimes.
2. **Run a separate lifetime-only `qkv_dbg` retention arm with normal PDL.** This directly tests the shared property of the successful recomputation probes.
3. If neither resolves it, run output retention and then the FULL-graph math-only-glue/original-cache-writer arm.

Use Booster for the quality comparisons; use the login GPU for capture/replay correctness and ownership inspection. I would put `--enforce-eager` and racecheck later: eager simultaneously changes compilation, capture, and the cache-writing path, while ordinary racecheck primarily targets shared-memory hazards and cannot clear a valid-address global-memory reuse race.
