# FlashMLA sparse decode: token packing (design review only)

2026-10-11. Idea: the 4 MTP3 tokens of a sequence attend over overlapping
top-k sets, but each token's CTA gathers and dequantizes its own keys. Pack
them: one CTA (M = 256) or a cluster of 4 CTAs sharing the dequant over DSMEM,
over the union of the 4 key sets with per-token masks. No code written.

`astra-prompt.md` is the design and our measurements (the full FlashMLA
`sparse_fp8` source at vllm-project/FlashMLA a8f794d was inlined after it);
`astra-review.md` is gpt-6-astra's answer. Its verdict:

- Producer (gather/dequant) and consumers (QK -> softmax -> PV) overlap on two
  K buffers, so block time ~ max(P, C). Packing cuts P by 4 but stretches C to
  the union, and union lengths round to whole 64-key blocks (r = 1.3 over 512
  live keys = 6 blocks on the slowest split, vs 4 with early stop).
- M = 256 or 128 per CTA does not fit (registers, smem). A cluster of 4 is
  plausible but needs 216 KB of DSMEM traffic per block, a new producer
  mapping, 8-arrival release barriers, token separated from head index, new
  planner metadata, and all 32 clusters resident at once.
- All-masked rows mid-sequence are already safe (finite MAX_INIT_VAL).
- Expected: zero to a few tenths of a ms beyond balanced early stop, with
  regression risk.
- Next step if pursued: a constant-fill build (no gather/dequant, everything
  else kept) and the per-block slope over a few block counts. Above ~3.3 us
  per block, packing cannot win at r = 1.3. Balanced early stop (re-planned
  splits, vllm-project/FlashMLA #27/#28) ranks first.
