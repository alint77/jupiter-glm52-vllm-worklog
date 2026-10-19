# Phase A: upstream ports from the 2026-10-19 scan (2026-10-19)

Two ports from `../2026-10-19-upstream-scan/`, served-tested on worktree pf11
(44153148ba + `phaseA-pf11.diff`), ABBA against prod (pf6) on 4 holds
(2407115-8) at the c=4 launcher config, dec1/dec4 harness (n decoders at ~5K
and ~50K global context, nsys windows). `launch_ab.sh`, `pa_compare.py`.

## A1: native next_n=4 DSA indexer decode on SM90 (#52164)

Our indexer flattened MTP3's 4 query rows per request into 4 single-row
requests (SM90 was thought to support only next_n 1/2); DeepGEMM's SM90
kernel does next_n=4 with a 2-CTA multicast cluster (metadata sized
num_sms/2). Ported as upstream: `native_next_n_supported`, schedule slots,
`_supports_native_decode`, per-step legality gate, metadata buffer narrowed.
Same op, same math: logits bitwise identical, top-k sets identical
(`bench_indexer_next_n.py`, every case below). Upstream tests pass (12/12).

Kernel microbench (Booster, CUDA graph + profiler), native vs flattened:

| M | 5K | 12.5K | 50K | 100K | 200K ctx/rank |
|---|--:|--:|--:|--:|--:|
| 4 | 3.14 / 3.04 | 3.23 / 4.19 | 4.74 / 6.94 | 6.91 / 9.54 | 11.2 / 16.0 |
| 8 | 3.20 / 3.58 | 4.06 / 4.80 | 6.94 / 9.47 | 11.1 / 15.9 | 24.7 / 48.0 |
| 16 | 3.94 / 4.51 | 4.96 / 6.14 | 11.1 / 15.8 | 23.8 / 40.5 | 50.0 / 120.3 |

(us per call, native / flat.) Served, per verify step (21 anchor calls),
paged-MQA-logits kernel time n - a, two holds each:

| arm | prod logits/step | n - a (holds) | mean |
|---|--:|--:|--:|
| dec1 ~5K, M=4 | 76-77 us | +6.0, +4.5 | +5.2 us |
| dec1 ~50K, M=4 | 104 us | -14.4, -14.9 | -14.7 us |
| dec4 ~5K, M=16 | 83 us | +4.6, +5.5 | +5.0 us |
| dec4 ~50K, M=16 | 172 us | -33.4, -32.5 | -32.9 us |

No launch-count difference per step. Step span/rest moves +-0.1-0.3 ms
between holds (noise, as expected for a <0.2% change). The native kernel is
~0.25 us/call slower below ~8K tokens per rank and faster above; real c=4
agentic traffic runs at ~290-330K global context (~75K/rank, see the
anchor-offload dump), where the microbench gives ~55 us/step at M=4 and
~350 us/step at M=16 (1.4% of a ~25 ms step), growing to ~1.5 ms at 200K/rank.
MTP acceptance within the arms' spread (2.8-3.6 both).

Caveat: native requires uniform decode lengths (`require_uniform`); padded
FULL-graph batches (a request with decode_len 0) take the pack/unpack path.
At c=4 MTP3 the decode sizes 4/8/12/16 are capture sizes, so no padding; at
c=8 (20-28 tokens padded up) check for `pack_seq`/`unpack_seq` kernels.

## A2: HBM accounting fixes

- #56902 applies: FlashMLA sparse kept `q_concat_buffer` views from its
  `__init__` reservation, pinning the first 72 MiB workspace after the shared
  workspace grew (72 -> 202 MiB from the indexer, -> 768 MiB from the MoE
  modular kernel). Ported: keep the specs, `get_simultaneous` per call.
  Startup observed free HBM: prod 2.84 / 2.84 GiB, pf11 2.91 / 2.91 GiB
  (+72 MiB per rank, ~14 hot experts if the reserve is lowered by it).
  Safety: the V2 runner never locks the workspace, so growth after capture
  would leave graphs on freed storage (the leak was hiding that). All three
  resizes (VLLM_DEBUG_WORKSPACE=1) happen in the profiling run, before
  capture, in every arm; none afterwards.
- #59368 / #60450 / #60493 (CUDA-graph memory estimate): not applicable, the
  tiered planner sizes everything from capacity - reserve and ignores vLLM's
  KV estimate.
- #59367 (DCP top-k merge scratch): not applicable, our merge scratch is
  64 MiB.

Both shipped on tp-sliced-moe (two commits after 44153148ba).
