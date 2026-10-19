# Astra v3 review: disposition (kernel v3e)

| # | finding | disposition |
|---|---|---|
| P1 | Grid barrier can deadlock with an ordinary launch: blocks that have arrived spin and hold their SMs, so blocks that have not started may never get one | **Adopted (v3e).** `cudaLaunchCooperativeKernel`, grid capped at the co-resident count (computed once). Time unchanged: v3d 10.30 / 10.02 µs → v3e 10.14 / 9.70 µs (layer 1 / 10, M=16, Booster). Stress test (map launches beside 8192³ GEMMs on another stream): no hang |
| P1 | External invalidation must be serialized with lookup and sweep; an inverse audit does not prove the bytes are fresh; REL comes before the copies finish | **Integration contract.** The prefill invalidation scatter goes on the map's stream. FlashMLA waits for the map kernel to finish, not for REL. Freshness is covered by the byte-for-byte replay against the source, which rewrites each step's new-token rows |
| P2 | `ctl[7]`/`ctl[8]` are signed int32 accumulators: UB overflow after ~39 h | **Adopted.** Unsigned wrapping adds; they are telemetry only. Epoch bound corrected to 2^30 steps ≈ 249 days at 50 steps/s (not decades) |
| P2 | Wrapper preconditions: device, alignment, index range, aliasing | **Partly adopted.** The wrapper now checks device and 16 B alignment. Index range and non-aliasing are established upstream: indices come from the indexer/DCP conversion (< num_blocks × block_size), and the buffers are allocated separately |
| Q2/Q3/Q5 | Acquire/sync chain, `__ldcg` sweep, parity counters | Found sound; no change |
| Q7 | `gtime_after` creates no real dependency | Noted. These timers are diagnostic only; the end-to-end spans are what we report |
| Q7 | Optimizations: distributed eviction (0.5–1.2 µs), leader fences, smaller blocks, no modulo | **Not adopted.** The user's call: ~10 µs already fits the window from top-k merge to FlashMLA (18.8–20.5 µs), so another 2–4 µs would not show in decode |

v3e checks (login: real dumps for layers 1/10, eager + graph; adversarial
eager + graph; stress): 0 bad entries, overflow 0, no hang
(`logs/ahc-v3e-check-login.log`).
