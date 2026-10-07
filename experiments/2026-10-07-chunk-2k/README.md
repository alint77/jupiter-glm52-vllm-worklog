# Prefill chunk 2K vs 4K on the full-memory-stack prod (2026-10-07)

Question: does `MAX_NUM_BATCHED_TOKENS=2048` free enough HBM (smaller
activation peak, smaller MoE buffers) to be worth its prefill cost? serve.sh
prod default otherwise (skip-layer MLA KV on Grace, fp8 drafter KV + weights,
drafter KV on Grace). One fresh hold per arm via `launch.sh <chunk>:<reserve>[:g]`;
`g` adds a piecewise CUDA graph at the chunk size to CAPTURE_SIZES. `run.sh`
measures TTFT (20K new on 14K cached x3, 8K new on 150K cached x2, 60K
uncached x2), then runs the 388K long-context stress from the
[reserve sweep](../2026-10-07-reserve-sweep/README.md). Logs: `run-*.log`;
server logs and nvidia-smi in /e/fscratch/profound/naeimitabiei1/chunk-2k/.

| arm | hot / rank | free after startup (minimum) | TTFT 20K/14K | 8K/150K | 60K uncached | 388K stress |
|---|--:|--:|--:|--:|--:|---|
| **4K, reserve 4.7 (prod)** | **3,531** | 2.70-2.79 GiB (2.53) | **2.851 s** | **1.543 s** | **8.33 s** | 0 OOM |
| 2K, reserve 4.7 | 3,548 | 2.62-2.70 (1.76) | 3.128 | 1.777 | 9.17 | 0 OOM |
| 2K, reserve 4.0 | 3,581 | 1.95-2.05 (1.76) | 3.122 | 1.692 | 9.17 | 0 OOM |
| 2K, reserve 3.5 | (3,607) | 1.61 GB < 1.89 required | refuses to start | | | |
| 2K, reserve 3.0 | (3,630) | | CUDA OOM at load | | | |
| 2K + graph at 2048, reserve 4.7 | 3,548 | 2.26-2.34 (1.76) | 3.137 | 1.746 | 9.16 | 0 OOM |

Hot counts are the minimum over ranks. The EngineDeadError in the 2K/4.7
server log is run.sh's teardown SIGTERM (16:35:51, after "server alive").

- 2K lowers the startup minimum 2.53 -> 1.76 GiB, but its usable floor is
  reserve ~4.0: **+50 hot experts/rank (~1 GiB), not 2 GB**.
- Prefill is **~10% slower** on every shape (+0.27 s at the Claude Code
  shape, +0.15-0.23 s at 150K context, +0.84 s on 60K uncached).
- A CUDA graph at 2048 changes nothing (full chunks are GPU-bound) and costs
  ~0.4 GiB free.
- 50 more hot experts (~1.4%) is worth roughly 0.1-0.3 ms/step, ~0.03 s over a
  150-step reply, against +0.27 s of prefill per turn.

Decision: keep 4K chunks at reserve 4.7 (serve.sh unchanged).
