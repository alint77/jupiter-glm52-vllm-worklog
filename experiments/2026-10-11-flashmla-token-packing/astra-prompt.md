# Design review request: "token packing" for FlashMLA sparse fp8 decode (SM90) under MTP

You are reviewing a kernel design idea before any code is written. Do NOT run
any commands (your sandbox cannot run on this machine). Everything you need is
inlined below: our measurements, the serving context, the full source of the
kernel we would modify, and the specific questions at the end. Be concrete and
quantitative; disagree with the idea if the numbers say so. Say what you are
inferring vs what follows from the source.

## 1. Serving context

- Hardware: NVIDIA GH200 (H100 96 GB HBM3, 132 SMs, sm_90a), 4 GPUs per node
  on NVLink (no NVSwitch). Measured ~630 TFLOP/s achieved bf16 dense on large
  GEMMs; HBM ~3.6 TB/s achievable. Kernel work is assessed at base clock.
- Model: GLM-5.3 (DeepSeek-V3.2-style MLA + DSA sparse attention, "V32" KV
  format): fp8 KV records of 656 bytes per token (512 fp8 nope + 4 fp32 scales,
  one per 128-dim tile + 64 bf16 rope), d_qk = 576, d_v = 512, h_kv = 1
  (absorbed MLA), 64 query heads per rank after the DCP query gather, 78 layers.
  The DSA indexer selects top-2048 keys per query token. GLM runs the indexer
  on only 21 of the 78 layers; the other layers reuse the most recent index set.
- Parallelism: TP4 with DCP4 (decode context parallel): the KV cache is
  sharded across the 4 ranks with interleave size 1 (token t lives on rank
  t % 4). Each rank gathers all 64 heads' queries (one-shot all-gather), runs
  sparse attention over the keys IT owns, and the 4 partial outputs are merged
  with an LSE-weighted reduce-scatter. So per rank each query token has ~512
  live keys (the rank's share of its top-2048; measured 466-537, max 597 over
  248 windows). vLLM compacts the live keys to the front of a fixed-width index
  row and pads with -1. We already ship width 768 (12 blocks of 64) instead of
  2048; that measured -0.47 ms/step served.
- Speculative decoding: MTP3 (3 draft tokens), so each sequence verifies 4
  query tokens per step at consecutive positions p, p+1, p+2, p+3. At c16 (16
  concurrent sequences) the decode step has 64 query tokens per rank.
- Call shape today: q (b=1, s_q=64, h_q=64, 576) bf16, indices
  (1, 64, 768) int32, KV fp8 paged (page 64). FlashMLA's SM90 planner gives
  num_sm_parts = 132 / s_q / (h_q/64) = 2, so grid = (1, 64, 2) = 128 CTAs,
  one wave, each CTA does one (token, half of the 768-wide row) = 6 blocks of
  64 keys, then a split-KV combine kernel merges the 2 splits.
- Hard constraint: "same math only". We may not change the algorithm, the
  key selection, or the precision of any operand. Accumulation-order changes at
  fp32 (e.g. different split boundaries, a different key order inside a
  softmax) are accepted; we already accepted them for the width change (max abs
  diff 1.5e-5 vs the 2048 width).

## 2. Measurements (per layer, per rank, GH200)

- c16 (64 tokens), width 768, profiler: main kernel 38.8 us + combine 7.7 us
  = ~46.5 us/layer, x78 = ~3.6 ms of a ~51 ms decode step. Kernel resources
  (ncu/cuobjdump): 384 threads, ~226 KB dynamic smem, 1 CTA/SM.
- 8 tokens (c2), width 768, 16 sm parts (each CTA <= 1 block): main ~14.0 us,
  combine ~6.2 us (graph-replay bench, 78 layers in one graph).
- A linear fit over three configs (8, 32, 64 tokens) gives roughly
  main ~= 13 us fixed + 4.3 us per 64-key block per CTA. This is a fit, not a
  per-phase profile. We have NOT yet profiled the main kernel with ncu at c16.
- Our own arithmetic for one 64-key block in one CTA (64 heads):
  - QK: 64 x 64 x 576 MACs = 2.36 M FMA; PV: 64 x 512 x 64 = 2.10 M FMA;
    ~8.9 MFLOP per block. At ~4.3 us that is ~2.1 TFLOP/s per SM, ~270
    TFLOP/s across 128 CTAs: well below dense peak, so a single block is not
    tensor-throughput-bound, but the chain QK -> softmax -> PV is serial
    inside WG0 (it waits on its own PV before the next QK).
  - Gather: 64 keys x 656 B = 42 KB of scattered HBM reads per block per
    CTA (~1.25 TB/s aggregate across 128 CTAs at 4.3 us/block), dequantized by
    the 128-thread producer warpgroup into 64 x 576 bf16 = 73.7 KB of smem.
  - With width 768 and ~512 live, ~1/3 of the blocks are -1 padding that the
    kernel still gathers (remapped to a real address), dequantizes (zeroed) and
    runs through both GEMMs; masked only in the softmax.
- The combine (7.7 us) depends on the number of splits, not the per-split
  work.

## 3. The idea: token packing

The 4 MTP tokens of one sequence attend over 4 top-k sets that are probably
heavily overlapping (adjacent positions, same context; and on 57 of 78 layers
they reuse the same indexer output, so the 4 rows come from the same indexer
call on 4 adjacent query positions). Today each token's CTA gathers and
dequantizes its own ~512 keys independently, so a key selected by all 4 tokens
is fetched from HBM and dequantized 4 times, in 4 CTAs.

Proposal: process the 4 tokens of a sequence together over the UNION of their
live key sets (deduplicated), with a per-(token, key) validity mask. Masked
(token, key) pairs get -inf in the softmax, so they contribute exactly 0 to
that token's sum and output (exp2(-inf) = 0, and 0 * finite V = 0): the math
per token is unchanged except for fp32 accumulation order. Two variants:

A. One CTA per sequence with M = 256 rows (4 tokens x 64 heads), keys gathered
   and dequantized once. Our concern: the existing kernel holds the 64 x 512
   fp32 output accumulator split over two consumer warpgroups (64 x 256 each,
   128 fp32 regs/thread) plus P. M = 256 would need 8 such warpgroups or a
   smem accumulator. Probably infeasible as-is; maybe M = 128 (2 tokens) per
   CTA, two CTAs per sequence.

B. A thread-block cluster of 4 CTAs per sequence, one CTA per token (each
   keeps today's M = 64 rows, its own Q, its own rO, its own mask). Each CTA's
   producer gathers and dequantizes 1/4 of every union block and writes it into
   its own smem AND the 3 peers' smem via DSMEM st.async, signalling the peers'
   transaction barriers. This is the existing h_q = 128 path generalized: today
   with NUM_HEADS = 128 the kernel launches a cluster of 2 CTAs (one per 64-head
   half of the SAME token), and each CTA dequantizes half of each 64-key block
   and st.async's it to the peer (bar_k_remote_ready, bar_k_avail with 2
   arrivals, see the source). Variant B changes the cluster from "2 head halves
   of one token" to "4 tokens of one sequence" and the K block contents from
   "this token's indices" to "the sequence's union indices", with an extra
   per-CTA validity mask (is_kv_valid becomes per-token over the union block).

Per CTA, variant B trades:
- gather + dequant work: union/4 keys instead of own (if union ~= 1.3 x own,
  ~0.33x of today);
- MMA + softmax work: union keys instead of own keys (if union ~= 1.3 x own,
  1.3x of today);
- plus cluster synchronisation: every K buffer needs all 4 producers' parts
  before any consumer can start, and all 4 consumers' releases before any
  producer can overwrite it (the cluster runs at the speed of its slowest CTA
  per block).
So it only wins if a block today is mostly gather/dequant-bound rather than
bound by the serial QK -> softmax -> PV chain. We do not know which.

Scheduling under variant B at c16: 16 sequences x 4 CTAs = 64 CTAs per split;
with 2 splits, 128 CTAs (same as today), each doing union/2 keys. Clusters of 4
must be co-resident in a GPC (H100 GPCs have up to 18 SMs, some fewer; 132 SMs
-> cluster-4 occupancy must be checked with cudaOccupancyMaxActiveClusters).

Index preparation (outside the kernel, once per indexer call, i.e. 21 times per
step since 57 layers reuse indices): for each sequence, build the union of the
4 rows' live indices (sort+unique over <= 4 x ~600 ints, or a hash), pad to a
multiple of 64, and a bitmask (4 bits per union key, or per-token bool rows).
The per-sequence union length varies; the split plan would have to come from
union lengths (FlashMLA's planner already takes topk_length per row in the
MODEL1 path; V32 asserts it off on SM90).

Ordering detail for "same math": today each token's keys are in the indexer's
compacted order; in the union the order changes. Within a token the result
differs only by fp32 summation order (online softmax rescaling sequence and
the PV accumulation), which we accept.

Unknowns we have NOT measured:
1. The actual overlap between the 4 MTP tokens' per-rank live sets
   (|union| / mean |own|). If the indexer's top-2048 at positions p..p+3 share,
   say, 90% of keys, union ~= 1.2-1.3x own; if 50%, union ~= 2.5x and the
   idea is dead. For reference, the indexer is a separate light attention
   (DSA lightning indexer) whose scores for nearby positions should be
   correlated, but we have no data.
2. Whether a block today is producer-bound (gather/dequant) or consumer-bound.

## 4. Related upstream work (none implements packing)

- deepseek-ai/FlashMLA #200 (open): packs the selected KV per query row ahead
  of the kernel (contiguous instead of scattered gather); still one token per
  CTA.
- deepseek-ai/FlashMLA #216 (open): dynamic top-k on SM90.
- vllm-project/FlashMLA #27 (open): topk_length early stop for SM90 V3.2
  (H20, 64 rows: 174.9 -> 54.4 us kernel, 74.9 including re-planning).
- vllm-project/FlashMLA #28 (open): planner 2.1-3.6x faster.
- vllm-project/vllm #58985 (open): wires topk_length into DCP decode
  (-3.9 to -8.3% decode step at 16-32 requests, from the 2048 width).
Our estimate for cherry-picking #27 + #28 here: -0.67 ms (6 -> ~4 blocks per
CTA at c16) minus ~0.13-0.2 ms planning if planned once per index set, net
~0.45-0.5 ms/step at c16. Token packing would stack with early stop only
partly (it would also use lengths).

## 5. Questions

1. Is the bottleneck reasoning right? From the source below, estimate per
   64-key block: producer time (128 threads, 2 tokens per thread, 8 x 16 B
   fp8 + 1 x 16 B scale + 2 x 16 B rope loads per token, dequant, smem
   stores), consumer WG0 chain (QK m64n64k16 x 36, softmax, PV m64n256k16 x 4),
   WG1 PV, and the K double-buffer handshake. Which side should dominate at
   ~4.3 us/block, and what explains a ~13 us fixed cost (Q TMA load, cluster
   barrier init, the 64 x 512 fp32 o_accum store per split = 128 KB per CTA,
   launch)? What single cheap experiment would settle producer- vs
   consumer-bound? Our candidates: (a) run the existing NUM_HEADS = 128
   cluster-2 path at the same s_q and compare per-CTA block time (it halves
   dequant per CTA at the same MMA per CTA); (b) a patched build with the
   dequant/gather replaced by a constant fill; (c) a patched build with the
   GEMMs removed; (d) ncu source-level stall sampling. Which, and what result
   would mean what?
2. Variant A vs B vs something else. Is M = 256 (or 128) per CTA viable on
   SM90 with this pipeline (register budget: WG0 192, WG1 160, producer 152
   per thread today; rO 64x256 fp32 per consumer WG)? Is B's DSMEM broadcast
   to 3 peers (each producer writes 4 copies of 1/4 block) cheaper than each
   CTA dequantizing everything itself? DSMEM st.async bandwidth and the
   transaction-barrier counts for 4 peers: any pitfalls?
3. Cluster-4 scheduling on a 132-SM H100: how many cluster-4s can be resident
   with 226 KB smem/CTA (1 CTA/SM)? Would we lose SMs (e.g. 128 -> fewer
   co-schedulable) and fall into a second wave?
4. Expected gain as a function of overlap ratio r = |union| / |own| and the
   producer share f of today's block time. Give the formula and your best
   guess range. At what r does it break even?
5. Cheaper alternatives that capture most of the benefit under "same math":
   e.g. (i) skip -1 blocks / early stop (topk_length, #27); (ii) more K
   buffers or a deeper producer pipeline (NUM_K_BUFS = 2 today) so the
   producer's latency hides; (iii) overlapping WG0's next QK with its own PV;
   (iv) a pre-pass that dequantizes each sequence's union keys once into a
   bf16 scratch (L2-resident: 16 seqs x ~700 keys x 1152 B ~= 13 MB vs 50 MB L2)
   and lets the unchanged-structure kernel TMA-load bf16 tiles (gathering bf16
   rows instead of fp8 + dequant); (v) fewer splits / no combine at c16.
   Rank them by expected gain per unit of work.
6. Correctness risks: masked keys with -inf for some tokens of a union block
   where ALL 64 keys are masked for one token (cur_max = -inf -> exp2f(-inf -
   -inf) = NaN?). Look at scale_softmax and MAX_INIT_VAL in the source and tell
   us whether an all-masked block for one row is already safe (it can happen
   today only via -1 padding at the tail) and what changes when it can happen
   in the middle of the sequence.
7. Anything in the source that makes either variant harder than we think
   (TMA Q descriptor shape (h_q, d_qk, s_q, b), the 5D O TMA store, the
   scheduler metadata's per-partition request ranges, PDL, cluster dims tied
   to NUM_M_BLOCKS).

## 6. Source (vllm-project/FlashMLA @ a8f794d, csrc/sm90/decode/sparse_fp8/)

Note how vLLM calls it: q is (b=1, s_q=num_decode_tokens, 64, 576), so the
grid's y dimension is the token and z the split; each CTA processes request
range [begin_req_idx, end_req_idx] = [0, 0] with a block range from the
planner's DecodingSchedMeta. get_meta on SM90 returns
{num_sm_parts = max(num_sms / s_q / (h_q/64), 1), fixed_overhead_num_blocks = 5,
block_size_topk = 64}.

