# Is NCCL's protocol the prefill collective problem?

Status: **Complete. No — the protocol was never the issue, and the bandwidth
was never bad.**

## Why

The [production profile](../2026-08-05-prod-profile/README.md) reported prefill's
96 MiB all-reduce at 2,064 us, a 70 GB/s ring bus rate, on
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` — NCCL's low-latency protocol, whose
flits are half flag bytes, on a 96 MiB message. That reads as a tuning error.

An [A/B](../2026-08-05-prefill-comms-ab/README.md) setting `NCCL_PROTO=Simple`
in the shell changed nothing and left the kernel names unchanged, so it proved
nothing either way. This probe takes vLLM out of the picture: plain
`torch.distributed`, four ranks, the identical message size.

## Result

Median of 30 iterations after 10 warmups, one Booster node, job `1246032`:

| `NCCL_PROTO` | 96 MiB | busbw | 192 MiB | busbw |
| --- | ---: | ---: | ---: | ---: |
| unset | 514 us | 294 GB/s | 953 us | 317 GB/s |
| `Simple` | 520 us | 290 GB/s | 955 us | 316 GB/s |
| `LL128` | 557 us | 271 GB/s | 1,042 us | 290 GB/s |
| `LL` | 1,105 us | 137 GB/s | 2,163 us | 140 GB/s |

Two conclusions:

1. **`NCCL_PROTO` is honoured here.** Forcing `LL` costs 2.1x, so the variable
   reaches the communicator in a plain program. It did not reach vLLM's, which
   is a plumbing question, not a tuning one — and vLLM sets it from Python
   (`batch_invariant.py:975`) rather than trusting the environment.
2. **NCCL's default already picks the fast protocol.** Unset and `Simple` agree
   to 1%. There was no protocol win available at this size.

## What it means for production

The hardware does this collective in ~514 us. Production's *mean* is 2,064 us,
but its **median is 482-520 us** — the hardware optimum — with a p90 of
6.1-6.5 ms (see the A/B writeup for the per-rank distribution).

So the "70 GB/s" figure was averaging wait time into transfer time. The
collective is not slow; it spends most of its measured time **waiting for the
slowest rank**, and since the cross-rank spread of means is only 4.0%, the
straggler rotates rather than being a fixed rank.

**This retires bandwidth and protocol as prefill levers** and replaces them with
EP load balance, which is the quantity replica assignment already addresses at
decode — and which is gated off above 16 tokens.

## Unresolved

Production runs `RING_LL` kernels yet reaches a p50 matching *Simple* here,
while forced `LL` here is twice as slow. The two are not the same operating
point; channel count is the likely difference (production launches
`grid=(24,1,1)` with TP, DCP and EP communicators coexisting). Not explained,
and not load-bearing for the conclusion.

## Files

| File | What |
| --- | --- |
| `probe.py` | Standalone 4-rank all-reduce bandwidth probe |
| `job.sh` | Sweeps unset / LL / LL128 / Simple |
| `out-*.txt` | Per-arm JSON results |
| `dbg-*.txt` | `NCCL_DEBUG=INFO` output (NCCL's own INFO lines did not surface) |
