# Does Grace DRAM contention slow the prefill cold-prefetch copy?

2026-10-15. Question from SuperInfer (papers/superinfer_paper_tex): Grace DRAM
gives ~384 GB/s per NUMA node and is half-duplex, so prefill's host-KV traffic
(skip_host_uva: KV writes into Grace, gathers from Grace) could slow the cold
expert prefetch copy (H2D, ~24.3 GiB per >=512-token step). Answered from the
existing nsys-fresh prefill-sizes trace (prod c8d61506be + interleave,
`sliced-prof/nfr-sizes-2277153`, one pure prefill step per window: 512-4000
tokens, fresh and after a cached 60K prefix). No new run.

Tools: `copy_overlap.py` (each H2D copy >= 1 MiB: bandwidth vs the kernel
classes running during it), `copy_wait.py` (per MoE layer: idle time before its
route_kernel that ends when its slot copies finish, and the slack between copy
completion and MoE start). Outputs `an-*.txt`.

## Result: no exposure; the copy is hidden at every size

- MoE never waited on its copy: 0 us of copy wait over 75 layers x 4 GPUs in
  every window.
- Minimum slack (MoE start minus its copies done), device 0: fresh 512: 244 us
  (p10 291); @60K 512: 1120; fresh 1K: 621; @60K 1K: 1794; fresh 2K: 2383;
  @60K 2K: 3777; fresh 4K: 4816.
- Host-KV kernels (concat_and_cache_ds_mla, cp_gather*, indexer_k_quant_and_cache)
  almost never run during a copy: the copies run under the previous layer's MoE
  GEMMs. Copies they do overlap (>= 20%) run at ~290-305 GB/s vs ~380, but
  carry < 1.5% of a step's copy bytes.
- Copy bandwidth per step: 381-391 GB/s fresh, down to 349-357 GB/s at @60K 2K.
  The slow copies are the ones overlapping all-reduce / DCP comm (median
  280-324 GB/s). Cause not isolated (correlation in the trace), and it costs
  nothing while slack stays positive.

## Implications

- Lower priority for removing the Grace DRAM / copy contention; it is not on
  the critical path at the current 512 threshold.
- Tightest case is fresh 512 (244 us minimum slack of ~0.9 ms per-layer copy).
  A lower threshold (384) would likely expose it; consistent with keeping 512.
- The @60K 4000 window holds two steps (the second below the prefetch
  threshold), so the per-layer slack mapping is invalid there; the idle-gap
  test still finds no wait.
