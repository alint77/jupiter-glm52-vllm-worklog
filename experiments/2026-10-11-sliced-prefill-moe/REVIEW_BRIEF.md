# Sliced prefill MoE kernel: brief for review

## Context
GLM-5.3 W4A16 MoE (256 routed experts, top-8, hidden H = 6144) is served TP-sliced
on 4 GH200s: every GPU holds all 256 experts, each a 512-wide slice of the
intermediate dim (w13: 6144 -> 1024 [gate 512 | up 512], w2: 512 -> 6144),
Marlin INT4 layout with bf16 group-32 scales. All 8 routes of every token are
local, so each GPU runs 8T routes per layer for a T-token prefill chunk
(T <= 4096). Hot (191 experts) and cold (65, staged into an HBM slot) tiers:
both are in HBM when the kernel runs. Routing is very skewed: at T = 4096
there are 32K routes over 256 experts (mean 128 rows/expert, max 750-1200).
Units of at most 128 rows by expert: per class (rows/unit <= 16/32/64/96/128),
units and rows:

| T | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| 512 | 130-164 u / 0.8-1.1K r | 39-41 / 0.9K | 20-29 / 0.9-1.3K | 8-14 / 0.6-1K | 1-4 / 0.1-0.5K |
| 4096 | 35-47 / 0.3K | 31-32 / 0.8K | 31-42 / 1.5-2K | 92-140 / 7.6-11K | 169-196 / 19-22K |

## Kernel (tiered_prefill.cu, excerpts below)
route (1 CTA) -> gather (8T sorted rows, f16 under a pow2 row scale) -> w13
(5 class launches, widest first, PDL-chained) -> act (silu*up -> f16 + pow2
scale, in place) -> w2 (5 class launches) -> combine (fp32 weighted sum of a
token's 8 rows -> bf16). GEMM: W is the wgmma A operand decoded from Marlin
INT4 words in registers (exact f16 products), X is B (TMA 128B swizzle, tokens
on N). 256 W rows per CTA (four 64-row tiles), T tiles per warpgroup, 4/T
consumer warpgroups (+ producer WG with setmaxnreg 40/232 for NT >= 96).
Narrow classes (NT <= 32) scale per group in fp32; wide ones scale weights in f16.
Numerics must stay as they are (exact products, fp32 accumulation; bitwise
equal output is our main correctness check for restructurings that keep the
per-element K order).

## Measurements (one GH200, isolated bench, real routing, CUDA graph)
Floors: weights 1.36 GB -> 406 us at 3.35 TB/s; compute 8T x 2 x (6144 x 1024 + 512 x 6144) flop
at 630 TFLOP/s (we use this as the base-clock peak): 123 us at T=512, 982 us at T=4096.

Wall: T=512 620-652 us (1.5-1.6x floor), 1024 884, 2048 1416, 4096 2594-2608 (2.65x).

Timeline of one T=4096 call (start offset us, duration us):
```
route 0 48 | gather 48 231 | w13: 128 279 714, 96 877 524, 64 1308 160, 32 1327 189, 16 1408 172
act 1579 116 | w2: 128 1695 459, 96 2141 323, 64 2453 52, 32 2494 46, 16 2532 35 | combine 2566 132
```
T=512: route 9, gather 22, w13 31->364 (16-class 226 us, from 138), act 17,
w2 381->607 (16-class 84 us), combine 20.

ncu (base clock, kernels serialized), T=4096:
```
kernel      grid   regs  us     tensor%active  dram%  rdMB  wrMB  issue%  smact%  tensor%elapsed
gather      528    52    238    0              70     290   380   47      95      0
w13 128     2052   168   694    83             51     1380  35    33      85      71
w13 96      2052   168   463    77             52     951   23    38      82      64
w13 64      1024   244   130    53             27     138   3     36      51      28
w13 32      1024   255   113    30             27     122   3     44      50      15
w13 16      1024   167   116    20             28     131   2     50      41      8
act         528    44    124    1              17     67    16    53      97      1
w2 128      12312  168   510    41             25     293   219   29      94      39
w2 96       12312  168   351    38             23     195   124   32      92      35
w2 64       6144   244   54     38             30     57    8     36      81      31
w2 32/16                 45/35  23/17
combine     4096   33    134    0              84     403   47    26      96      0
route       1      28    52
```
T=512: w13 16-class 220 us (dram 56%, issue 59%, smact 78%), 32 113, 64 126; w2 16-class 97 us
(issue 58%, tensor 20%), 32/64 ~51 each, 128/96 ~43 each (tensor 19-23%).

## My reading
1. w2 (K = 512): CTA = 256 W rows x NT tokens x K 512 = only 4 stages at CK 128;
   prologue (barrier init + first TMA ~1-2 us), and the epilogue (acc -> smem
   bf16 -> 16 B global stores, 64 KB) are not overlapped (wide classes are
   1 CTA/SM). tensor 39%/active. Also X (NT x 512 f16, 128 KB at NT=128) is
   re-read by each of the 24 CTAs of a unit.
2. w13 wide classes are ok-ish per active cycle (77-83%) but lose to wave
   quantization / class serialization (PDL lets the next class start only when
   the last CTAs of the previous one are placed), then a ~180 us narrow-class
   tail: few narrow units, each CTA streams K=6144 alone at ~440 cycles per
   k16 step (one warpgroup, T=4, wait<0> per round), SMs half idle.
3. act_kernel uses 64 of 256 threads (inter=512 -> 64 16-byte vectors per row)
   with two __syncthreads per row: 124 us vs a ~30 us floor.
4. route_kernel is one CTA, latency-bound (dependent global loads per route):
   48 us at 32K routes.
5. gather writes 8T f16 rows (400 MB at 4K) and combine reads 8T bf16 rows
   (400 MB): 2 x ~130 us floors. Fusing gather into the w13 B loads (per-row
   TMA) and combine into the w2 epilogue (fp32 atomics: non-deterministic) are
   the big structural options.

## Plan (in order)
A. act_kernel: a warp (or 64 threads) per row, several rows per CTA, no block
   barriers (bitwise identical).
B. route_kernel: more ILP / several CTAs for the counting pass.
C. w2 restructure: CTA = (unit, range of B consecutive 256-row blocks), X tile
   (NT x 512) loaded once and kept resident in smem; W (+scales) streamed
   through a ring continuously across blocks by the producer; epilogue per
   block via a smem staging buffer and async 1D bulk stores (cp.async.bulk
   shared->global per 512 B output row) so consumers move on to the next
   block. smem at NT=128: X 128 KB + staging (128 x 528 B = 66 KB, or half and
   two passes) + W ring 3 x 18 KB. Same per-element K order -> bitwise equal.
D. Narrow-class latency: more warps per CTA (T=1, 4 consumer WGs) or 128-row
   CTAs for narrow classes, or deterministic split-K (two K halves, the last
   finisher adds p0 + p1 in fixed order) so the few narrow units at T=4096
   finish faster; and/or launch order changes.
E. Gather fused into w13 via per-row 2D TMA (box {64, 1}) of a T x H f16 copy
   of x, if TMA swizzle-128 boxes of height 1 land in the same layout as a full
   box (I believe the swizzle is a function of the smem address bits).
