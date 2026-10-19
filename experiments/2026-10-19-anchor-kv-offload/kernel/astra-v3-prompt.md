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


## astra-v1-disposition.md

# Astra v1 review: what v2 does with it

Review: `astra-v1-review.md` (gpt-6-astra, xhigh, static review of the v1
draft inlined in `astra-v1-prompt.md`).

| # | Finding | Disposition |
|---|---|---|
| 1 | Bounded victim search can find no victim with ample capacity; failure turns a valid entry into -1 | Adopted. Found the same bug independently (v1 fixed before the review landed). v2: one block sweeps for every claim in hand order, up to two laps (lap 1 clears ref bits, lap 2 must find every unprotected row). The wrapper requires n <= nrows, so hits + claims <= N and every claim gets a victim; the overflow counter stays as an assert. No gather fallback: the bound covers every shape we launch (M <= 16 x 768 = 12288 < 16384). |
| 2 | Plain concurrent accesses (duplicate hit stores, where[] load vs CAS, hand-wrap selectors) are data races in the CUDA C++ model | Adopted. v2 packs stamp and ref into one status word (epoch << 1 \| ref), so a hit is a single relaxed device-scope store. The where[] read in lookup and the new-slot invalidation are relaxed atomics. Selection runs in a single block, so the cross-selector races are gone. Conditional eviction (`where[old] == victim`) is added as defense. |
| 3 | Invalidation must clear both directions; enumerate every content mutation; allocation generations | Both directions: already the case in v1 lookup and kept. The full inventory of mutation paths (block reuse, spec rewrites, prefill writes) is an integration item. Generations are noted as an alternative to the per-slot scatter; decide at integration. |
| 4 | Phases are safe with stream order; FlashMLA must wait for the whole copy; don't reset counters from one CTA of copy | v2 resets counters in the *last* CTA of copy (done-counter pattern, after every CTA has read them). That addresses the concern without adding a launch. Lookup's grid barrier uses the same pattern before the sweep. |
| 5 | Keep `anchor_source` + slot-valued indices for `stager.stage`; hot indices only for FlashMLA; keep valid_counts and schedule | Adopted for integration (it is a real trap: stager groups are keyed by the anchor cache pointer). Pack the current token once and store it twice. |
| 6 | DCP reserved row N + j with the original j; padding all -1; pure-decode only, but invalidate on every other write path; MTP drafter reads | Adopted. The kernel already uses the original j (new_slots[j] = -1 for tokens other ranks own). Contracts go in the integration. |
| 7 | Epoch overflow; define counters; init; wrapper validation | Validation adopted (dtypes, sizes, contiguity, n <= nrows). Epoch: < 2^30 steps (decades at ~50 steps/s), documented and not handled. Counters 4/5/6 are cumulative, 2/3 per call. |
| 8 | 4 launches + atomics not shown to fit | Measured: v1 15.5-16.4 us. v2 is 2 launches; measure on Booster against the main-branch overlap. |
| 9 | 3 phases; FIFO via partitioned scan; drop pending list | v2 selection *is* a block scan over a window of candidates (CLOCK keeps the ~1-1.5 pp over FIFO for one extra bit in the same word). The pending list stays for now; trying a full remap in copy is next if copy's tail shows up. |
| 10 | Simpler baseline (gather all distinct rows each call); HBM accounting | Baseline: ~2.3 MB per anchor per call is ~6 us of C2C alone, on top of the stager bookkeeping we measured at 9-13 us. Rejected as the target but kept as a reference point. Accounting adopted: 2.69 GiB was 22 layers; 21 anchors = 2.57 GiB; minus 215 MiB buffers, ~16 MiB where[] and ~2 MiB row state, that frees ~2.33 GiB/rank. |


## anchor_hot.cu (v3d, current)

```cuda
   1  // Anchor hot cache (v3d): an HBM buffer of N rows in front of an anchor
   2  // layer's fp8_ds_mla main cache on Grace. Per decode step and anchor, the
   3  // converted top-k (rank-local slots) becomes buffer rows in one launch, one
   4  // thread per entry:
   5  //   lookup  hits stamp their row and set its CLOCK reference bit (one packed
   6  //           status word, epoch << 1 | ref); misses claim their slot (where[s]:
   7  //           -1 -> -2, one winner per slot) and are listed; this step's tokens
   8  //           map to reserved rows N + j and their slots are invalidated (the
   9  //           KV write changes the slot's content).
  10  //   sweep   the last block to arrive gives every claim a victim in CLOCK hand
  11  //           order (block scan over a window of candidates: a row read this
  12  //           step is skipped, a set reference bit is cleared and passed),
  13  //           publishes where[s], advances the step and releases the others.
  14  //           Every block has started once the last one arrives, so waiting
  15  //           blocks never hold an SM a pending block needs.
  16  //   copy    the other blocks: once the last block arrives the claim list is
  17  //           final, so they read the missed rows from Grace (16 B words over
  18  //           their threads) while it sweeps, and store them to their rows
  19  //           after the release; each entry that missed reads its row.
  20  // Entries keep position, multiplicity and -1, so FlashMLA reads the same
  21  // bytes in the same order as from the full cache.
  22  #include <cub/block/block_scan.cuh>
  23  #include <cuda/atomic>
  24  #include <cuda_runtime.h>
  25  #include <stdint.h>
  26  #include <torch/extension.h>
  27  #include <c10/cuda/CUDAStream.h>
  28  
  29  namespace {
  30  
  31  constexpr int ROW_WORDS = 656 / 16;  // 41 x 16 B
  32  constexpr int THREADS = 1024;
  33  constexpr int SWEEP_V = 4;     // candidates per thread per sweep round
  34  constexpr int SH_MISS = 4096;  // claims the sweep stages in shared memory
  35  
  36  // ctl: [0] step (epoch; status holds it << 1, so < 2^30 steps), [1] CLOCK
  37  // hand, [2 + 2 * (step & 1)] claims of the call at that step, [6] overflow
  38  // (claim without a victim: must stay 0), [7] candidates swept, [8] claims so
  39  // far, [9] blocks arrived, [10..15] and [24..26] AHC_PROF timers,
  40  // [16] released step, [20] step whose claim list is final, [21] its sweeping
  41  // block.
  42  constexpr int CTL = 32;
  43  constexpr int REL = 16;
  44  constexpr int ARR = 20;
  45  constexpr int LASTB = 21;
  46  
  47  struct State {
  48    int32_t* where;     // [S] slot -> row, -1 absent, -2 claimed this step
  49    int32_t* slot_of;   // [N] row -> slot, -1 free
  50    int32_t* status;    // [N] (step last read or filled) << 1 | CLOCK ref bit
  51    int32_t* ctl;
  52    int32_t* miss;      // [MAXE] claimed slots
  53    int32_t* miss_row;  // [MAXE] their rows
  54  };
  55  
  56  // Accesses that race with other threads of the launch.
  57  __device__ __forceinline__ int ld_rlx(const int32_t* p) {
  58    return cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*const_cast<int32_t*>(p))
  59        .load(cuda::memory_order_relaxed);
  60  }
  61  __device__ __forceinline__ void st_rlx(int32_t* p, int32_t v) {
  62    cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*p).store(
  63        v, cuda::memory_order_relaxed);
  64  }
  65  
  66  __device__ __forceinline__ int warp_append(int32_t* counter, bool want) {
  67    unsigned m = __ballot_sync(0xffffffffu, want);
  68    if (!m) return -1;
  69    int lane = threadIdx.x & 31;
  70    int leader = __ffs(m) - 1;
  71    int base = 0;
  72    if (lane == leader) base = atomicAdd(counter, __popc(m));
  73    base = __shfl_sync(0xffffffffu, base, leader);
  74    return want ? base + __popc(m & ((1u << lane) - 1)) : -1;
  75  }
  76  
  77  __device__ __forceinline__ int64_t gtime() {
  78    int64_t t;
  79    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  80    return t;
  81  }
  82  // read the timer only once dep is available
  83  __device__ __forceinline__ int64_t gtime_after(int dep) {
  84    int64_t t;
  85    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : "r"(dep));
  86    return t;
  87  }
  88  
  89  // One block gives every claim a victim, in hand order. Candidate p of a round
  90  // is row (hand + p) % N; the hand stops right after the candidate taken by
  91  // the last claim, so exactly the rows it passed lose their reference bit.
  92  // Everything a candidate needs is loaded before the scan; after it, stores.
  93  // The other blocks wait at the barrier, so plain (L2) loads see final values.
  94  __device__ void sweep(State st, int step, int nrows, int nmiss, int hand) {
  95    using Scan = cub::BlockScan<int, THREADS>;
  96    __shared__ typename Scan::TempStorage scan;
  97    __shared__ int sh_stop, sh_taken;
  98    __shared__ int32_t sh_miss[SH_MISS];
  99  #ifdef AHC_PROF
 100    int64_t t_scan = 0, t_enter = gtime();
 101  #endif
 102  #ifdef AHC_PROF
 103    int64_t t_pre = gtime(), t_cand = 0;
 104  #endif
 105    int assigned = 0;
 106    const int max_rounds = 2 * (nrows / (THREADS * SWEEP_V) + 1) + 1;
 107    for (int round = 0; assigned < nmiss && round < max_rounds; ++round) {
 108      const int need = nmiss - assigned;
 109      // thread t owns candidates p = t * V + v (contiguous, scan in hand order)
 110      int c[SWEEP_V], stv[SWEEP_V], old[SWEEP_V];
 111      bool ok[SWEEP_V], refd[SWEEP_V];
 112  #pragma unroll
 113      for (int v = 0; v < SWEEP_V; ++v) {
 114        c[v] = (hand + threadIdx.x * SWEEP_V + v) % nrows;
 115        stv[v] = __ldcg(&st.status[c[v]]);
 116        old[v] = __ldcg(&st.slot_of[c[v]]);
 117      }
 118      if (round == 0) {
 119        for (int i = threadIdx.x; i < min(nmiss, SH_MISS); i += THREADS)
 120          sh_miss[i] = __ldcg(&st.miss[i]);
 121      }
 122      int cnt = 0;
 123  #pragma unroll
 124      for (int v = 0; v < SWEEP_V; ++v) {
 125        bool fresh = (stv[v] >> 1) != step;
 126        refd[v] = fresh && (stv[v] & 1);
 127        ok[v] = fresh && !refd[v];
 128        cnt += ok[v];
 129      }
 130  #ifdef AHC_PROF
 131      if (round == 0) t_cand = gtime_after(cnt + old[0] + old[SWEEP_V - 1]);
 132  #endif
 133      int off, total;
 134      Scan(scan).ExclusiveSum(cnt, off, total);
 135  #ifdef AHC_PROF
 136      if (round == 0 && threadIdx.x == 0) t_scan = gtime();
 137  #endif
 138      if (threadIdx.x == 0) {
 139        sh_stop = THREADS * SWEEP_V;
 140        sh_taken = min(total, need);
 141      }
 142      __syncthreads();
 143      if (total >= need && off < need && off + cnt >= need) {
 144        int r = off;
 145  #pragma unroll
 146        for (int v = 0; v < SWEEP_V; ++v) {
 147          r += ok[v];
 148          if (ok[v] && r == need) sh_stop = threadIdx.x * SWEEP_V + v + 1;
 149        }
 150      }
 151      __syncthreads();
 152      const int stop = sh_stop;
 153      int r = off;
 154  #pragma unroll
 155      for (int v = 0; v < SWEEP_V; ++v) {
 156        int p = threadIdx.x * SWEEP_V + v;
 157        if (p >= stop) break;
 158        if (refd[v]) st.status[c[v]] = stv[v] & ~1;
 159        if (ok[v]) {
 160          int i = assigned + r;
 161          ++r;
 162          int s = i < SH_MISS ? sh_miss[i] : __ldcg(&st.miss[i]);
 163          // where/slot_of stay inverse (every writer clears both), so the
 164          // victim's slot maps to it
 165          if (old[v] >= 0) st.where[old[v]] = -1;
 166          st.slot_of[c[v]] = s;
 167          st.status[c[v]] = step << 1;
 168          st.where[s] = c[v];
 169          st.miss_row[i] = c[v];
 170        }
 171      }
 172      hand = (hand + stop) % nrows;
 173      assigned += sh_taken;
 174      if (threadIdx.x == 0) st.ctl[7] += stop;
 175  #ifdef AHC_PROF
 176      if (threadIdx.x == 0) st.ctl[14] += 1;
 177  #endif
 178      __syncthreads();
 179    }
 180    for (int i = assigned + threadIdx.x; i < nmiss; i += THREADS) {
 181      st.where[i < SH_MISS ? sh_miss[i] : __ldcg(&st.miss[i])] = -1;
 182      st.miss_row[i] = -1;
 183    }
 184    if (threadIdx.x == 0) {
 185      st.ctl[1] = hand;
 186      st.ctl[6] += nmiss - assigned;
 187      st.ctl[8] += nmiss;
 188  #ifdef AHC_PROF
 189      st.ctl[12] += (int32_t)(t_scan ? gtime() - t_scan : 0);
 190      st.ctl[24] += (int32_t)(t_pre - t_enter);
 191      st.ctl[25] += (int32_t)(t_cand - t_pre);
 192      st.ctl[26] += (int32_t)(t_scan - t_cand);
 193  #endif
 194    }
 195  }
 196  
 197  template <int UNROLL>
 198  __global__ void __launch_bounds__(THREADS)
 199  map_kernel(State st, const int32_t* __restrict__ idx, int n,
 200             const int32_t* __restrict__ new_slots, int t, int32_t* __restrict__ out,
 201             const int4* __restrict__ src, int4* __restrict__ dst, int nrows) {
 202    __shared__ int32_t sh_new[64];
 203    __shared__ bool last;
 204    if (threadIdx.x < 64) sh_new[threadIdx.x] = threadIdx.x < t ? new_slots[threadIdx.x] : -1;
 205    __syncthreads();
 206    const int step = st.ctl[0] + 1;
 207    const int hand = st.ctl[1];  // written only by the previous launch's sweep
 208    int32_t* cnt = st.ctl + 2 + 2 * (step & 1);
 209  #ifdef AHC_PROF
 210    const int64_t t_start = gtime();
 211    if (threadIdx.x == 0) atomicMin((unsigned long long*)(st.ctl + 10), (unsigned long long)t_start);
 212  #endif
 213    if (blockIdx.x == 0 && threadIdx.x < t) {
 214      // no entry reads where[] for these slots (they map to reserved rows)
 215      int s = sh_new[threadIdx.x];
 216      if (s >= 0) {
 217        int r = ld_rlx(&st.where[s]);
 218        if (r >= 0) {
 219          st_rlx(&st.where[s], -1);
 220          st_rlx(&st.slot_of[r], -1);
 221          st_rlx(&st.status[r], ld_rlx(&st.status[r]) & ~1);
 222        }
 223      }
 224    }
 225    const int e = blockIdx.x * THREADS + threadIdx.x;
 226    const int s = e < n ? idx[e] : -1;
 227    bool pend = false, claim = false;
 228    if (s >= 0) {
 229      int j = -1;
 230      for (int k = 0; k < t; ++k) j = (sh_new[k] == s) ? k : j;
 231      int o;
 232      if (j >= 0) {
 233        o = nrows + j;
 234      } else {
 235        o = ld_rlx(&st.where[s]);
 236        if (o >= 0) {
 237          st_rlx(&st.status[o], (step << 1) | 1);
 238        } else {
 239          pend = true;
 240          if (o == -1) claim = atomicCAS(&st.where[s], -1, -2) == -1;
 241        }
 242      }
 243      if (!pend) out[e] = o;
 244    } else if (e < n) {
 245      out[e] = -1;
 246    }
 247    int m = warp_append(&cnt[0], claim);
 248    if (claim) st.miss[m] = s;
 249  
 250    // grid barrier: the last block to arrive sweeps, the others wait for it
 251    __threadfence();
 252    __syncthreads();
 253    if (threadIdx.x == 0) last = atomicAdd(&st.ctl[9], 1) == (int)gridDim.x - 1;
 254    __syncthreads();
 255    cuda::atomic_ref<int32_t, cuda::thread_scope_device> arr(st.ctl[ARR]), rel(st.ctl[REL]);
 256    if (last) {
 257      __threadfence();
 258      if (threadIdx.x == 0) {
 259        st.ctl[LASTB] = blockIdx.x;
 260        arr.store(step, cuda::memory_order_release);
 261      }
 262  #ifdef AHC_PROF
 263      const int64_t t_last = gtime();
 264  #endif
 265      sweep(st, step, nrows, __ldcg(&cnt[0]), hand);
 266      __threadfence();
 267      __syncthreads();
 268      if (threadIdx.x == 0) {
 269        // the previous call (stream-ordered before this launch) is done
 270        st.ctl[2 + 2 * ((step + 1) & 1)] = 0;
 271        st.ctl[9] = 0;
 272        st.ctl[0] = step;
 273  #ifdef AHC_PROF
 274        int64_t t0 = (int64_t)*(unsigned long long*)(st.ctl + 10);
 275        st.ctl[13] += (int32_t)(t_last - t0);
 276        st.ctl[15] += (int32_t)(gtime() - t_last);
 277        *(unsigned long long*)(st.ctl + 10) = ~0ull;
 278  #endif
 279        rel.store(step, cuda::memory_order_release);
 280      }
 281    } else if (threadIdx.x == 0) {
 282      while (arr.load(cuda::memory_order_acquire) != step) __nanosleep(32);
 283    }
 284    __syncthreads();
 285  
 286    // claim list final: the copiers (all blocks but the sweeping one, unless it
 287    // is alone) read their words from Grace now, store them after the release
 288    const int nmiss = __ldcg(&cnt[0]);
 289    const int lastb = __ldcg(&st.ctl[LASTB]);
 290    const bool alone = gridDim.x == 1;
 291    const bool copier = alone || !last;
 292    const int ncp = alone ? 1 : gridDim.x - 1;
 293    const int cb = alone || (int)blockIdx.x < lastb ? blockIdx.x : blockIdx.x - 1;
 294    const int ce = copier ? cb * THREADS + threadIdx.x : 0;
 295    const int stride = ncp * THREADS;
 296    const int total = copier ? nmiss * ROW_WORDS : 0;
 297    int4 pre[UNROLL];
 298  #pragma unroll
 299    for (int u = 0; u < UNROLL; ++u) {
 300      int f = ce + u * stride;
 301      if (f < total) {
 302        int i = f / ROW_WORDS;
 303        pre[u] = src[(int64_t)__ldcg(&st.miss[i]) * ROW_WORDS + (f - i * ROW_WORDS)];
 304      }
 305    }
 306    if (!last && threadIdx.x == 0) {
 307      while (rel.load(cuda::memory_order_acquire) != step) __nanosleep(32);
 308    }
 309    __syncthreads();
 310  
 311    if (pend) out[e] = __ldcg(&st.where[s]);
 312  #pragma unroll
 313    for (int u = 0; u < UNROLL; ++u) {
 314      int f = ce + u * stride;
 315      if (f < total) {
 316        int i = f / ROW_WORDS;
 317        int r = __ldcg(&st.miss_row[i]);
 318        if (r >= 0) dst[(int64_t)r * ROW_WORDS + (f - i * ROW_WORDS)] = pre[u];
 319      }
 320    }
 321    for (int f0 = ce + UNROLL * stride; f0 < total; f0 += stride * UNROLL) {
 322      int4 v[UNROLL];
 323      int64_t to[UNROLL];
 324  #pragma unroll
 325      for (int u = 0; u < UNROLL; ++u) {
 326        int f = f0 + u * stride;
 327        to[u] = -1;
 328        if (f < total) {
 329          int i = f / ROW_WORDS, w = f - i * ROW_WORDS;
 330          int r = __ldcg(&st.miss_row[i]);
 331          if (r >= 0) {
 332            v[u] = src[(int64_t)__ldcg(&st.miss[i]) * ROW_WORDS + w];
 333            to[u] = (int64_t)r * ROW_WORDS + w;
 334          }
 335        }
 336      }
 337  #pragma unroll
 338      for (int u = 0; u < UNROLL; ++u)
 339        if (to[u] >= 0) dst[to[u]] = v[u];
 340    }
 341  }
 342  
 343  }  // namespace
 344  
 345  void anchor_hot_map(torch::Tensor idx, torch::Tensor new_slots, torch::Tensor out,
 346                      torch::Tensor src, torch::Tensor dst, torch::Tensor where,
 347                      torch::Tensor slot_of, torch::Tensor status, torch::Tensor ctl,
 348                      torch::Tensor miss, torch::Tensor miss_row) {
 349    for (auto* x : {&idx, &new_slots, &out, &where, &slot_of, &status, &ctl, &miss, &miss_row}) {
 350      TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->scalar_type() == at::kInt);
 351    }
 352    TORCH_CHECK(dst.is_cuda() && dst.is_contiguous() && src.is_contiguous());
 353    const int64_t n = idx.numel();
 354    const int t = new_slots.numel();
 355    const int nrows = slot_of.numel();
 356    const int64_t nslots = where.numel();
 357    // A sweep window holds distinct rows. Every claim gets a victim within two
 358    // laps when hit rows + claims <= nrows, which n <= nrows guarantees.
 359    TORCH_CHECK(t <= 64 && n <= nrows && nrows >= THREADS * SWEEP_V);
 360    TORCH_CHECK(out.numel() == n && miss.numel() >= n && miss_row.numel() >= n &&
 361                status.numel() == nrows && ctl.numel() >= CTL);
 362    TORCH_CHECK(src.nbytes() >= nslots * ROW_WORDS * 16 &&
 363                dst.nbytes() >= (int64_t)(nrows + 64) * ROW_WORDS * 16);
 364    TORCH_CHECK(nslots < (1ll << 31));
 365    State st{where.data_ptr<int32_t>(), slot_of.data_ptr<int32_t>(),
 366             status.data_ptr<int32_t>(), ctl.data_ptr<int32_t>(),
 367             miss.data_ptr<int32_t>(), miss_row.data_ptr<int32_t>()};
 368    auto stream = at::cuda::getCurrentCUDAStream();
 369    const int blocks = std::max<int>((n + THREADS - 1) / THREADS, 1);
 370    map_kernel<4><<<blocks, THREADS, 0, stream>>>(
 371        st, idx.data_ptr<int32_t>(), n, new_slots.data_ptr<int32_t>(), t,
 372        out.data_ptr<int32_t>(), reinterpret_cast<const int4*>(src.data_ptr()),
 373        reinterpret_cast<int4*>(dst.data_ptr()), nrows);
 374  }
 375  
 376  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("anchor_hot_map", &anchor_hot_map); }
```

Review statically (no commands needed). Rank findings by severity: correctness first (races, ordering, deadlock, wrong bytes), then integration traps, then performance with concrete expected gains.
