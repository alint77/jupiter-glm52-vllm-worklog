# Prefill: nsys breakdown and levers

2026-10-11. MTP3 TP-sliced c4 config (`claude-glm53-sliced-mtp3-c4.sh`,
`max_num_batched_tokens` 4096, DCP4), tree `fmla-p1`. `prof_prefill.py` runs
three windows, each twice unprofiled and once under `/start_profile`:

1. **fresh16k**: 16K new tokens, no cache (4 chunks).
2. **delta8k@60k**: a cached 60K prefix plus 8K new (2 chunks plus a
   313-token tail).
3. **mixed**: three decoders at 10K context producing 3000 tokens each, plus a
   delta8k@60k prefill. Each prefill chunk then shares a step with decode
   tokens.

Two c4 nodes (2272603, 2272604) and one c1 node (2272605). The traces are in
`/e/fscratch/profound/$USER/sliced-prof/pfn-*`. `pf_groups.py` gives the
per-device kernel groups. The c1 mixed TTFT (46 s) is queueing
(max_num_seqs=1); ignore it.

## Timings (unprofiled, c4)

| window | TTFT | prefill rate |
| --- | ---: | ---: |
| fresh16k | 2.27-2.28 s | ~7.0K tok/s |
| delta8k@60k | 1.53-1.68 s | ~5.1K tok/s |
| mixed (prefill) | 2.23 s | ~3.7K tok/s; decoders 113-115 tok/s |

## Per 4096-token chunk (device 0; all ranks within ~5%)

| group | pure chunk (ms) | mixed step (ms) |
| --- | ---: | ---: |
| step span | **547-594** | **884-906** |
| MoE (tiered_prefill gather/gemm/act/combine) | 230-244 | 233-244 |
| attention kernel | 50-63 (sparse_attn_fwd) | **272-299 (decode kernel)** |
| DCP q all-gather (NCCL ring) | 75-83 | 74-80 |
| DCP lse_combine (one-shot) | 47 | 48 |
| dense GEMMs (nvjet) | 69-79 | 72-77 |
| TP all-reduce x2 per layer | 41-56 | 40-49 |
| barrier / elementwise / other | ~50-77 | ~150 |

The 313-token tail chunk takes 141 ms, of which **MoE is 137 ms**.

Rank skew is small: NCCL mean minus min across ranks is 6-26 ms per window.
The collectives are transfer time, not waiting. Per layer, the q all-gather is
0.77 ms (~300 GB/s bus bandwidth). The all-reduce is ~0.27 ms.

Compute floor per chunk: ~180 ms (MoE ~80, attention projections ~48,
sparse attention ~36, indexer). We are at 3.1x that for pure chunks and
4.9x for mixed steps.

## Findings

1. **Mixed steps are about +320 ms (+60%) per chunk.**
   - With decoders present, `q.shape[0] != num_actual_tokens`, so the prefill
     tokens go through the decode kernel `flash_fwd_splitkv_mla_fp8_sparse`.
   - Skip-KV staging is off when `num_prefills > 0`. The 57 skip layers
     therefore read their KV from Grace per (token, top-k) entry: 4.3-4.6 ms
     per layer, against 1.17 ms on the 21 anchor layers (HBM KV) and about
     0.7 ms for the prefill kernel.
   - In the agent swarm nearly every chunk is mixed. This is the UX case.
2. **The MoE has a large fixed cost per chunk.**
   - 313 tokens take 137 ms and 4096 tokens take ~240 ms.
   - At 313 x 8 routes, every expert is touched, so each chunk reads all
     weights. That includes the 4910 cold experts per rank (~334 MiB per
     layer, ~25 GiB per chunk). 137 ms for this is ~180 GB/s, consistent with
     a C2C stream that is not overlapped with compute.
   - Only ~100 ms of the 240 ms scales with tokens.
3. **DCP attention communication is ~125 ms per chunk (22%).**
   - The q all-gather moves 4096 x 64 heads x 576 bf16 to every rank (~226 MB
     received per layer).
   - The one-shot LSE combine reads ~200 MB of partial outputs per layer.
   - The KV shard is only 10 MB at 16K and 40 MB at 60K (fp8).
4. **Dense GEMMs** are 75 ms against a ~48 ms floor. The TP all-reduce
   (~45 ms) is near NVLink bandwidth.

## Levers, ranked by served impact (same math in each)

1. **Mixed steps through the prefill path.** Split the batch: prefill tokens
   take gather and upconvert, then `sparse_attn_fwd` (KV read once per
   layer); decode tokens keep the decode kernel with staged KV. Expected:
   895 → ~600 ms per mixed 4K step (-30%). Largest for the swarm.
2. **MoE cold stream.** Overlap the cold-expert C2C stream with hot-expert
   compute (prefetch cold weights into an HBM slot on a side stream, as the
   decode path does), or raise its rate (~180 GB/s now). Up to ~100 ms per
   chunk. Probe first: time MoE vs chunk size and hot/cold split in isolation.
3. **Bigger chunks when no decoder is waiting** (pure prefill): 8192 tokens
   amortizes the fixed MoE cost, ~-12% per 8K tokens. It costs ITL in mixed
   steps, so make it adaptive.
4. **Sequence-split DCP prefill.** All-gather the KV shard (10-40 MB). Use an
   all-to-all for q to 1/4 of the tokens x 64 heads (~56 MB) and back for the
   output (~50 MB). No LSE combine. This keeps the h=64 prefill kernel.
   About -70 to -90 ms per chunk.
5. Dense GEMM tiles (~25 ms per chunk).
