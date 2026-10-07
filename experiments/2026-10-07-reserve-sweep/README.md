# HBM reserve sweep for the full-memory-stack prod default (2026-10-07)

serve.sh default (skip-layer MLA KV on Grace, fp8 drafter KV + weights, drafter
KV on Grace), one fresh hold per value (holds 2211389-2211393), `sweep.sh`
running `../2026-10-02-ingraph-prefetch/stress_long.sh` with
STRESS_MAX_CTX=390000: 60K-token prompt, +2-8K per turn, +40K uncached every
10th turn; 42 turns to 388K tokens, 16 output tokens each. Logs: `run-*.log`,
server logs and per-second nvidia-smi in /e/fscratch/profound/naeimitabiei1/
ingraph-prefetch/{server,mem}-reserve-*.

| RESERVE_GB | hot / rank | free after startup (>= 2.53 GiB) | 388K stress | peak used |
|--:|--:|--:|---|--:|
| 5.14 | 3,510 | 3.10-3.16 GiB | 0 OOM | 95.8 GiB |
| 4.9 | 3,522 | 2.86-2.94 | 0 OOM | 96.1 |
| **4.7** | **3,531** | 2.68-2.77 | 0 OOM | 96.3 |
| 4.55 | 3,538 | 2.54-2.62 | 0 OOM | 96.4 |
| 4.4 | (3,545) | 2.58-2.64 GB < 2.71 required | refuses to start | |

The binding constraint is the startup check (measured activation peak + 1 GiB),
not the long-context stress: every value that starts survives 388K. Default
set to 4.7: 4.55 cleared the check by 0.01 GiB on one rank, and the check is
fail-closed, so node-to-node variation could keep prod from starting.
