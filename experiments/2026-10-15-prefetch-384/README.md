# Cold-prefetch threshold 384 vs 512

2026-10-15. `VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS` 384 vs the serve.sh
default 512, c4 launcher config (sliced MTP3, interleaved prefill), vLLM
c8d61506be (frozen worktree nsys2). Four holds in parallel (`launch.sh`), one
per arm: nsys windows (`arm.sh sizes`, prof_sizes.py) and un-profiled TTFT
(`arm.sh ttft`, `ttft_sizes.py`, 8 reps interleaved) per threshold. Pure
prefill steps (max_tokens 1, no decoders) at 320/384/448/512 requested tokens,
fresh and after a cached 60K prefix. Jobs 2327608-11, logs in `logs/`, reports
in `sliced-prof/pf{384,512}-{sizes,ttft}-*`.

## The gate sees the padded size

The token gate runs on `x.shape[0]`, which inside a captured prefill graph is
the capture size (…, 256, 384, 512, …). Prompts carry ~30 template tokens, and
@60K steps also recompute the uncached block remainder, so in the nsys windows
(device 0, H2D copies >= 1 MiB):

| window | real step | padded | 512: prefetch | 384: prefetch |
|---|---|---|---|---|
| fresh320 | ~350 | 384 | no | yes (24.3 GiB) |
| every other case | 385-512 | 512 | yes | yes |

So 512 already covered real steps of 385-512 tokens; 384 adds 257-384. Only
fresh320 differs between arms; the other rows measure node offset.

## Result: -4 to -5 ms (~5%) on 257-384-token steps, copy still hidden

GPU span of the step, nsys, device 0 (ms):

| case | 384 | 512 | delta |
|---|---|---|---|
| fresh320 (differs) | 100.3 | 105.2 | -4.9 |
| fresh384 | 104.1 | 105.3 | -1.2 |
| fresh448 | 106.1 | 107.3 | -1.2 |
| fresh512 | 111.4 | 112.0 | -0.6 |

Un-profiled TTFT median (ms, 8 reps):

| case | 384 | 512 | delta |
|---|---|---|---|
| fresh320 (differs) | 98.6 | 106.8 | -8.2 |
| fresh384 | 103.4 | 106.7 | -3.3 |
| fresh448 | 104.3 | 106.1 | -1.9 |
| fresh512 | 113.0 | 115.0 | -2.0 |

The @60K rows (all prefetched in both arms) move by -9 to -19 ms in the
TTFT pair and -2 to -14 in the sizes pair: node offset, larger and noisier at
60K context, so they are not evidence either way.

Copy exposure at the padded-384 step (`copy_wait.py`, fresh320, 384 arm): MoE
waited on its copy 0 us over 75 layers on all 4 GPUs; minimum slack 162-180 us
(median 186-236 us); 24.3 GiB at 362-368 GB/s. Hidden, with less margin than
at 512 (187-201 us at the padded-512 step here, 244 us in 2026-10-15-c2c-copy-contention).

## Shipped

serve.sh default 512 -> 384. Safe for every launcher: the load-time check needs
it above max_num_seqs x verify tokens (16 x 4 = 64 at c16). The env var is in
the compile hash, so each cache root recompiles once (~5 min) on its next
start.

Next below: 256 would add 129-256-token steps (padded 256), where per-layer
compute is about half; slack would likely go negative. Not tested.
