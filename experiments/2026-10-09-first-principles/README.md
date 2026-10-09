# First principles: where the remaining time can go (2026-10-09)

Cheap offline probes behind the idea list (agentic capture held-out steps,
prod profile at 3,670 hot, c=1 8-token steps unless noted).

- `tp_vs_ep.py`: MoE per step, slowest GPU as served (cost table + deployed
  replica balancer) **9.15 ms**, mean GPU 7.71. Every expert sliced 4-way on
  its intermediate dim (each GPU reads 1/4 of every touched expert; the
  existing post-MoE all-reduce sums the partials): 7.12 ms at no per-slice
  overhead, 8.29 at +0.5 us per extra slice, 9.45 at +1.0. Touched per layer
  (all GPUs): 37.8 hot, 3.6 cold.
- `cold_cache.py`: per (layer, GPU) LRU of recently read cold experts in place
  of the S least-touched hot ones, no replicas in either arm. Cold per
  GPU-layer slowest / mean: S=0 1.559 / 0.708; S=1 1.410 / 0.615; S=2 1.307 /
  0.555; S=4 1.196 / 0.494; S=8 1.102 / 0.444.
- `int4_entropy.py`: int4 codes 3.708 bits order-0, 3.707 conditioned on the
  previous code along K (7.3% lossless headroom); bf16 group scales 7.1 bits
  of 16 (3.1% of expert bytes).
- `../2026-10-09-c8-profile/kernels.py --outside` (DFlash2 k=3, 1x5K): outside
  the verify graph ~1.7 ms = two 465 MB lm_head passes (143 us each, at the
  HBM floor), a top-p sort / scan / softmax over 8 x 151K (195 + 56 + 41 us),
  the fp8 drafter's GEMMs (0.26 ms), its attention and decode_gemm (0.3 ms).
