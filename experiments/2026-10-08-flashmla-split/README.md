# FlashMLA sparse decode under DCP4: split count and index width

From ../2026-10-08-decode-dive-3 #3: per layer the fp8 sparse decode costs
~18 us main + ~8 us split-KV combine for ~512 live keys per rank.

## Source (vllm-project/FlashMLA a8f794d, `csrc/api/sparse_decode.h`,
## `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`)

- sm90 parts: `num_sms / s_q / (h_q/64)` = 132 / 8 / 1 = **16**; grid
  (NUM_M_BLOCKS, s_q, num_sm_parts) = the trace's [1, 8, 16]. Not tunable
  without patching FlashMLA.
- A -1 index is not skipped: it is remapped to row 63 of block 0, loaded,
  zeroed, dequantized, run through both GEMMs and masked only in the softmax.
  Prod passes 2048-wide indices with this rank's ~512 compacted to the front,
  so ~3/4 of the main kernel's tiles are padding.
- `topk_length` (the kernel's early stop) asserts on sm90 for the V3.2 KV
  format: "V3.2 does not support dynamic topk length".

## Bench (`bench.py`, `launch.sh`; 78 layers in one CUDA graph, prod shape
## q (1, 8, 64, 576), shared schedule metadata per step; GH200, 1 GPU)

us per layer (graph replay; main / combine from the profiler, which adds
overhead so they do not sum to the replay time):

| width | 5K ctx (1280 owned) | 100K ctx (25000 owned) | main | combine | max abs diff vs 2048 |
|---|---|---|---|---|---|
| 2048 (prod) | 21.37 | 21.57 | 17.3 | 7.9 | 0 |
| 1024 | 16.99 | 17.44 | 13.8-14.1 | 6.5-6.8 | 1.5e-5 |
| 768 | 16.75 | 16.75 | 14.0 | 6.2 | 1.5e-5 |
| 640 | 15.68 | 15.44 | 11.6 | 7.8-8.1 | 1.5e-5 |

Live keys per query were 466-537. Differences are fp32 accumulation order
(different split boundaries), at bf16 rounding.

Per step (78 layers): 1024 saves ~4.1-4.4 us/layer = **~0.33 ms**, 640 ~5.8
us = **~0.45 ms** (less than the 0.5-0.9 estimated: the main kernel has a
large fixed part, and the combine stays ~6-8 us because the split count
stays 16).

A row whose live count exceeds the width would silently drop keys. With
`cp_kv_cache_interleave_size = 1` (prod) the per-rank count of a top-2048
set is ~512 +- ~20; 1024 needs one rank to own half the selection.

## Served A/B (`chain_dw.sh`, 4 nodes x 4 arms alternating, vllm bfa5e0d112)

dw0 = full width, dw1 = `VLLM_DCP_SPARSE_DECODE_WIDTH=768`; prod serve.sh
otherwise (`KINDS=dw0,dw1 compare_ba.py`, +- is one standard error):

| | full width | 768 |
|---|---|---|
| agentic decode, 274 requests | | **-0.472 +- 0.027 ms/step** |
| same, all 276 requests | | -0.383 +- 0.278 |
| 50K / 130K decode, 64 requests | | **-0.478 +- 0.021 ms/step** |
| GSM8K (8 x 200) | 0.913 | 0.919 |
| hot / startup free / 388K stress | 3670 / 3.46 / 8 of 8 OK | same |

The two excluded requests are short (20-23 steps), one per arm (dw0-2227046-2
at 72 ms/step, dw1-2227048-2 at 120); cause not found. Overflow monitor
(`live-slots-per-window.txt`, 248 windows after each rank's startup window):
max live slots per row 597, 0 rows over 768. The startup window itself counts
capture / warm-up rows with placeholder indices (up to 2048 on one rank) and
should be ignored.
