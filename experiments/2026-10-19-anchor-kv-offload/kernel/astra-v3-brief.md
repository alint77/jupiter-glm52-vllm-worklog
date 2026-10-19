# Anchor hot cache v3: review brief (follow-up to your v1 review)

Context is unchanged from the v1 brief (GLM-5.3 DSA/MLA, fp8_ds_mla 656 B
rows, TP4 + DCP4, MTP3, c=4, FULL CUDA graphs; goal: anchors' main KV on
Grace, an HBM hot buffer of N=16384 rows + 64 reserved rows per anchor layer,
FlashMLA must read identical bytes in identical order). Your v1 findings and
what was done with each are in the disposition table below. The kernel is
now one launch; the full source follows.

## Measurements (Booster GH200, one GPU, NUMA-bound, C2C probe 421 GB/s)

Replay of a real top-k dump (c=4 agentic traffic, ~300K-token contexts,
M=16 query tokens per step, 12288 entries/step of which ~8300 valid,
~3558 distinct rows), CUDA graph per shape, torch profiler, 5000 steps.
Correctness: every entry checked byte for byte against the Grace source
(-1 -> -1, new token j -> N + j, else the hot row holds the source row),
overflow counter 0, where/slot_of inverse audit, on login (eager + graph,
layers 1/10/20) and Booster. All versions: 0 bad entries.

| version | launches | layer 1 M=16 span | layer 10 |
|---|--:|--:|--:|
| v1 (lookup, select, copy, finish) | 4 | 15.46 us | 16.42 us |
| v2 (last-block select in lookup, last-block finish in copy) | 2 | 15.65 | 14.91 |
| v2b (finish moved to lookup's last block, parity counters) | 2 | 14.21 | 14.01 |
| v3 (one kernel, last block sweeps, others spin on a release flag, copy in all blocks) | 1 | 14.30 | 13.50 |
| v3b (plain __ldcg in the sweep instead of relaxed atomics) | 1 | 14.05 | 13.57 |
| v3c (no where[old] check: 4096 random loads from one SM cost ~2 us) | 1 | 11.49 | 11.07 |
| v3d (copy prefetch from Grace during the sweep; sweeping block excluded from the copy) | 1 | 10.30 (p99 12.26) | 10.02 (p99 12.23) |

v3d timers, layer 1 / 10: lookup body 3.9 / 3.6 us, sweep 4.2 / 3.9 us
(candidate loads 1.2, scan 0.4, stores 1.8 / 1.5), remainder ~2.1 us. A first
v3d build crashed: the sweeping block's copy index went negative when it was
block 0 (f < total held for negative f with total = 0); fixed by ce = 0 for
non-copiers, memcheck clean.

Claims/step: layer 1 303, layer 10 235. In-kernel timers (globaltimer, last
block) for v3c layer 10: first block start -> last block arrives 3.4 us;
sweep 4.2 us = candidate loads 1.4 + block scan 0.4 + stores 1.5 + fence and
release ~0.7; rest of the kernel ~3.4 us (copy after release). The sweep
always finishes in one round (4096 candidates; ~840 passed per step).

## Disposition of your v1 findings

(see astra-v1-disposition.md, summarized) 1 adopted (complete two-lap sweep,
n <= nrows bound); 2 adopted (packed status word, relaxed atomics where
accesses race in lookup, single-block selection); conditional eviction was
adopted in v2 and then removed in v3c: it cost ~2 us (4096 random where[]
loads from one SM), and the inverse invariant is maintained by every writer
and audited instead; 3 both-direction invalidation kept, the rest is
integration; 4 counters reset in the sweeping block, which every block has
reached (no CTA can still need them); 5-6 integration contracts adopted;
7 validation adopted, epoch < 2^30 documented; 9 one launch instead of
three; 10 HBM accounting corrected (~2.33 GiB/rank freed).

## Questions

1. The grid barrier: blocks arrive on a counter; the last arriver sweeps
   and publishes (release) a step-valued flag; the others spin (acquire,
   __nanosleep) on it. I claim this cannot deadlock under any co-scheduling
   (every block has started by the time the last one arrives; spinners only
   exist after arriving) even when other streams' kernels occupy SMs. Is
   that right? Any MPS/green-context/preemption/graph case that breaks it?
2. Memory ordering: thread 0 acquires the flag, then __syncthreads, then
   other threads read with __ldcg (sweep outputs written by the sweeping
   block's threads, each of which ran __threadfence before the
   __syncthreads that precedes the release store). Is that sufficient in
   the PTX memory model, or do the readers need their own acquire? Same for
   the arrival side (all threads __threadfence, __syncthreads, thread 0
   atomicAdd; the last block's threads __threadfence after reading `last`).
3. The sweep reads with plain __ldcg after the barrier (no concurrent
   writers then). Any problem with L1 staleness on the sweeping SM (it ran
   lookup work on the same arrays: where[] loads/CAS, status stores)?
4. Unconditional eviction relies on: where[s] = r <=> slot_of[r] = s for
   r >= 0, maintained by lookup's new-slot invalidation (clears both), the
   sweep (evict old slot, set both for the claim), and the future external
   invalidation scatter (must clear both). Anything that can break it?
5. Parity counters (claims count at ctl[2 + 2*(step&1)], the sweeping block
   zeroes the other parity's, i.e. the previous launch's) and ctl[0]/hand
   read with plain loads at kernel start (written only by the previous
   launch). Sound across graph replays of different shapes interleaved with
   eager calls on one stream?
6. v3d: the other blocks read the missed rows from Grace (src, pinned host
   memory, read through UVA) as soon as the claim list is final (a second
   flag published by the sweeping block right after it knows it is last),
   and store after the release. Is reading src for claim i before its
   destination row is known safe in every case (overflow -> miss_row = -1
   skips the store)? Anything else that writes src concurrently in the
   planned integration (the current step's KV write goes to slots that are
   never claimed, since they map to reserved rows)?
7. Where would you look for the next 2-4 us? The remaining costs are the
   lookup body (~3.4 us for one entry per thread: idx load, where load,
   status store or CAS, warp-aggregated append), the single-block sweep
   (~4 us; ~1.5 us of it is ~1200 scattered stores from one SM), and the
   post-release copy. Options I see: defer slot_of/status/where[old]
   stores and ref clears to all blocks after the release (breaks the
   multi-round/wrap case unless kept in-sweep there); a two-level sweep
   across blocks (needs a second barrier); PDL so the launch overlaps the
   producer's tail; smaller blocks for the lookup.
