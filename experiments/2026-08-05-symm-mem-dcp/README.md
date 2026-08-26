# Runtime qualification of the DCP symmetric-memory guard

Status: **Complete, with a data-collection bug that cost one arm.** The hang
`990b1d378` guards against did not reproduce in three runs, and per-rank window
registration is symmetric, so the divergence hypothesis behind the guard is
unsupported at this configuration. **The guard was not removed** — three clean
runs do not refute a race — and all instrumentation was reverted from the
source tree.

## Why this exists

[Round 9 of the DCP port](../2026-07-18-dcp-port/README.md) hit a hang: job
`976497` stalled with one rank inside `ncclCommWindowRegister` while the others
had already entered the matching all-reduce. NCCL documents window registration
as a *collective* operation, and the reasoning at the time was that DCP can give
ranks different allocator histories even when the current collective has a
uniform shape.

Commit `990b1d378` ("Avoid NCCL symmetric all-reduce under DCP") therefore
disabled symmetric-memory **all-reduce** whenever
`decode_context_parallel_size > 1`, leaving the NVLS all-gather/reduce-scatter
path available. Its own writeup says the split "passed commit hooks but still
needs one runtime qualification", and HANDOFF has carried that debt since
2026-07-19. This is that qualification.

It matters because A2A+NVLS reached **190.15 tok/s** once against the qualified
`ag_rs` default's 179.56 — a 5.9% result parked as experimental purely on this
reliability question.

## Method

Two temporary source changes, both since reverted (`symm-mem-instrumentation.patch`):

- `pynccl_allocator.py` — one INFO line per `ncclCommWindowRegister`, carrying
  rank, window index, address and size. Without this the registrations are
  invisible and any divergence is unobservable.
- `all_reduce_utils.py` + `envs.py` — `VLLM_NCCL_SYMM_MEM_ALLOW_DCP`, an escape
  hatch that lifts the guard so the disabled path can actually be exercised.

Three runs on the production c4 config (MTP3, DCP4, exact replica assignment,
V2 runner), with `VLLM_USE_NCCL_SYMM_MEM=1` and the guard lifted: job
`1246565`, then jobs `1246656` / `1246657` as an `ag_rs` vs `a2a` pair.

Each run does a greedy smoke completion and then eight 16K prompts at
concurrency 4 from the Phase 33 prompt set, so the allocator sees the mixed
shapes DCP actually produces. A hang would itself have been the result; the job
is written to log and continue rather than block.

## Result

The guard was genuinely lifted — every run logs
`Using ['NCCL_SYMM_MEM', 'CUSTOM', 'PYNCCL']` for `tp:0` and
`['NCCL_SYMM_MEM', 'PYNCCL']` for `dcp:0` and `ep:0`, i.e. symmetric memory
first in dispatch order on the DCP group.

**No run hung.** All three came up, returned byte-identical greedy text
(`" Paris. Distance from Paris to Lyon is"`), and completed the benchmark.

Per-rank window registrations, recomputed from the server logs:

| run | arm | TP0 | TP1 | TP2 | TP3 |
| --- | --- | ---: | ---: | ---: | ---: |
| `1246565` | (pre-pair) | 20 | 20 | 20 | 20 |
| `1246656`/`1246657` | `ag_rs` | 25 | 25 | 25 | 25 |
| " | `a2a` | **0** | **0** | **0** | **0** |

**Registration is exactly symmetric on every path that registers at all.** Ranks
agree on count, on window index, and on size: 96 MiB windows for the activation
buffers, 2 MiB for the small ones, and a 512 MiB / 128 MiB pair. There is no
sign of the divergent allocator history the guard assumes.

**The `a2a` arm registered no windows at all.** It started at the same second as
`ag_rs` with the same environment and the same backend dispatch order, served
correct text, and finished the benchmark — but `ncclCommWindowRegister` was
never called. So under `--dcp-comm-backend a2a`, symmetric memory is inert
rather than dangerous, which also means **this run says nothing about the
A2A+NVLS configuration that originally hung**. That is the configuration the
190.15 tok/s number came from, and it remains unqualified.

Also logged on every rank, unchanged by any arm:
`SymmMemCommunicator: symmetric memory multicast operations are not supported`.

## The collection bug

`job.sh` greps `${arm}-server.err` for the registration lines. They go to
`${arm}-server.out`. The consequence is that all three saved
`*-registrations.txt` files are **byte-identical** (`md5 892a0ecc…`) — each is
just a re-dump of the first run's `symm-server.out`, and the arm-specific data
was never written to them. The per-rank counts printed in the slurm logs
(20/20/20/20) are from the same wrong source and are *not* the arms' counts.

The table above was rebuilt from the raw `*-server.out` files:

```bash
grep -ah "registering window" ag_rs-server.out \
  | sed 's/.*\(Worker_TP[0-9]\).*/\1/' | sort | uniq -c
```

Had the `a2a` arm's zero not been visible in the raw logs, this experiment would
have reported "symmetric registration on both backends" — a false positive
produced by an arm that never ran the code path. The saved artefacts are the
wrong ones; the server logs are the record.

## Verdict

| | |
| --- | --- |
| Hang reproduced | No, in 3 runs |
| Registration divergence across ranks | None observed |
| Guard `990b1d378` | **Kept** |
| A2A+NVLS (the config that hung) | **Still unqualified** — no windows registered under `a2a` here |

Three clean runs are weak evidence against a race, and the arm that would have
tested the original failure did not exercise the path. Removing the guard on
this basis would trade a 5.9% throughput result against a hang risk that has
been observed once and explained mechanically. Not taken.

The next useful measurement is `a2a` with NVLS actually engaged — first
establishing *why* the `a2a` path never registers a window, since that is
currently the whole reason the arm is uninformative.

## Files

| File | What |
| --- | --- |
| `job.sh` | One arm per invocation, `ARM=ag_rs\|a2a` |
| `symm-mem-instrumentation.patch` | The reverted source changes |
| `{a2a,ag_rs}-server.out` | **The real record** — registrations land here |
| `{a2a,ag_rs}-registrations.txt`, `registrations.txt` | Byte-identical, wrong source; kept only to document the bug |
| `*-semantic.json` | Greedy smoke output per arm |
