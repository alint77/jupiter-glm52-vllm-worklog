# Astra integration review (ahc-b1 = e9d65055e3): disposition

Verdict: the stream ordering (KV write → side-stream map → main FlashMLA →
next map) is sound; no demonstrated race. Points:

| finding | disposition |
|---|---|
| `record_stream(main)` on side-stream conversion outputs is right; also record the stager stream's read of `indices` (now protected only transitively) | Agreed. Not a bug in this config. Add `indices.record_stream(self.stream)` in `SkipKVStager.stage` before shipping (not in the frozen A/B tree) |
| P1? External KV writers (connectors, swap/offload restore, block copy) bypass `do_kv_cache_update` | None are enabled here (no kv_transfer_config, no CPU offload or swap in v1; the MTP drafter writes only layer 78, which has no hot cache). To do: refuse `VLLM_ANCHOR_HOT_CACHE` when a KV connector is configured |
| P1? Map and update predicates are not literally the same (`md[0]`, `fp8_use_mixed_batch`, prefix vs full slot_mapping) | Under DCP the mixed-batch path is the only one, `md[0]` is the target's metadata under speculative decoding, and padding is -1. To do: per-call checks (`numel <= max_entries`, `t <= 64`) and assert that an update which skips invalidation is followed by a map |
| Invalidation leaves `status` untouched | Checked against the kernel: a row with `slot_of = -1` is skipped on eviction (`old < 0`), and its stale stamp only delays its reuse by one sweep |
| FlashMLA cache size / non-scheduled path | The served path is the scheduled one (DCP4 + VLLM_DCP_DECODE_SCHED). Check mode asserts it. The in-situ check below covers it bitwise, including the reserved rows that are re-quantized by `concat_and_cache_mla` |
| Need a Grace-vs-Grace determinism control | Not needed for this result: nondeterminism could only cause false mismatches, never hide one, and we saw 0 mismatches |
| Host counter: non-atomic adds, unsynchronized snapshot | The adds are serialized on main. Treat the log line as telemetry; the final reading was stable over 30 s |
| `"same"` needs an assertion that the retained charge covers the hot caches | The hot caches are 0.24 GiB vs a 2.57 GiB retained charge at the 800K pool. To do: assert |
| Initialize before capture; scope the singleton; recompute the reserved map per anchor | It works as is (extension and tables are built at the first forward, before capture). Before shipping: move the build to post-KV-allocation init |

In-situ check (hold 2412782, `launch_check.sh`, dec4 at c=4 config,
`VLLM_ANCHOR_HOT_CACHE=same`, `VLLM_ANCHOR_HOT_CACHE_CHECK=1`): ~11,357
mapped anchor calls per rank (~540 verify steps × 21 anchors) at ~5K and
~50K context, **0 mismatching output/LSE elements** against FlashMLA on the
Grace cache. Planner: 14,284 hot / 4,916 cold experts per rank, the same as
prod. Observed free HBM after startup 5.14 GiB vs prod 2.84–2.91 GiB
(+2.3 GiB, as planned).
