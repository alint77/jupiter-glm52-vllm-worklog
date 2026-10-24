# Anchor indexer fork/join (2026-10-24) — shelved

Idea: in FULL verify graphs, run the anchor's DSA indexer on a side stream. The k projection forks at the layer entry; the rest forks after q_c. Both overlap the main q path (q_b, q RoPE, KV write, W_UK, DCP q gather). Joins sit at the hot cache's convert/map stream (start_mqa) and in forward_mqa. Gated by `VLLM_ANCHOR_INDEXER_FORK`; same ops, no math change. Motivation is in `../2026-10-24-verify-cycle/`: the indexer is fully serial in the anchor, which runs 344 us against 257 for a skip layer.

Code is local only, not pushed:
- `worktrees/afork` (branch `anchor-indexer-fork`): 0cb6fdeaca, on 23d2796ed5.
- `worktrees/afork2` (branch `anchor-indexer-fork-v2`):
  - c6c01b1989 fixes the one-shot collective race.
  - c3f61a3bee adds hardening from the review.

## Findings

1. **Collective race in v1.** The indexer's DCP candidate all-gather and the main stream's q `all_gather_cat` are both one-shot kernels on the custom all-reduce's `Signal`. `barrier_at_start` reads and stores `_flag[blockIdx.x]` (`csrc/custom_all_reduce.cuh:202-213`), so running the two at once races on the flags, which means a hang or an early barrier.
   - v2 makes the main q gather wait on an event recorded right after the indexer's all-gather.
   - The ordering costs about nothing: the side chain through its all-gather (~45 us) is longer than the main q path before the gather (~17 us).
2. **Astra review** (`astra/fork-review.md`, gpt-6-astra xhigh): no remaining hazard on the DCP4 path. It confirmed:
   - the top-k write-after-read ordering against the previous anchor's shared layers and the skip-KV stager;
   - `record_stream` lifetimes;
   - rank ordering.

   It also raised conditional issues, fixed in c3f61a3bee:
   - The head-major gather was outside the fence.
   - `forward_mqa` took the shared workspace for the q concat before the join. The views alias at offset 0, so this would race on DCP1.
   - The fork did not require q_lora.
   - The test compared sums of indices.

   Inductor reordering or functionalization of the opaque ops was left to the trace to check.
3. **Test:** `tests/model_executor/test_indexer_fork.py` on the login GPU (correctness only). It passes. Both negative controls fail:
   - without the gather fence, 16/145 elements mismatch;
   - without the final join, capture fails with "unjoined work".
4. **HBM: the reason the work is shelved.**
   - Served arm af1f (hold 2492777, v1, fork on) failed at startup with `Tiered MoE observed free HBM is below the runtime reserve: 1566310400 bytes available, 2712141824 required`.
   - Graph capture took **2.71 GiB, against 1.46 GiB for prod** (prodg-dec4-2489367): **+1.25 GiB per GPU**.
   - Likely cause, not verified: blocks used across streams inside a capture (`record_stream`, per-stream pools) are not reused until the capture ends, so each of the 21 anchors keeps its indexer intermediates live.

## Estimate vs cost

The critical path from the glue to FlashMLA, from the 50K trace (`../2026-10-24-verify-cycle/an/cycle-2489367-w2.txt`):

| | serial (prod) | fork v2 |
|---|--:|--:|
| glue → FlashMLA start, per anchor | ~97 us | ~76 us |

- That is ~21 us per anchor × 21 anchors ≈ **0.45 ms of a 22.4 ms verify step (~2%)**. This was estimated, never measured.
- Holding that requires giving back ~1.25 GiB of HBM per GPU, against a measured slack of ~0.47 GiB. The alternative, moving more experts to Grace, costs more than 2%.
- Recovering the memory would need restructured lifetimes: keep references until the join instead of `record_stream`, or use persistent indexer buffers. That is uncertain work for a 2% gain.

## Files

- `launch_af.sh <hold> <run> <kind>:<arm> ...`: arms a = fork off, f = fork on; kinds are dec4 (nsys) and probe. The worktree is selected with `WT`.
- `astra/fork-prompt.md`, `astra/fork-review.md`
- `logs/`; the server logs are in `/e/fscratch/profound/naeimitabiei1/sliced-prof/af1f-dec4-2492777/`.
