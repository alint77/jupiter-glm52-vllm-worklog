# c=4 agent swarm, whole engine: timeline, steps, DRAM / VRAM

2026-10-18. Prod c=4 config (`claude-glm53-sliced-mtp3-c4.sh`: sliced MTP3,
4 seqs, KV pool 2 x 400K, reserve 3.6, interleave + SRPT, fastokens, prefix
fix), frozen tree `worktrees/pf5` (vLLM 144f2f4aee). Ten full-node holds at
once (2390683-91, 2390719), own compile / Triton caches seeded from the pf5
c=4 cache. Replay as `../2026-10-15-c4-timeline/`: 4 closed-loop agents, 48
consecutive Claude Code turns each (cc-replay.jsonl, session 898193f4,
offsets 0/48/96/144), streamed `/v1/messages`, max_tokens 2048, zero think
time, 900 s.

| arm | hold | what |
|---|---|---|
| tlpyspy | 2390683 | timeline, py-spy on API server / EngineCore / worker 0 whole run |
| tlmem | 2390684 | timeline, no profiler (variance, py-spy overhead check) |
| tlnsys A | 2390685 | timeline, node-traced nsys windows 6 s at 30/120/240/.../720 s |
| tlnsys B | 2390691 | same, windows at 60/180/.../780 s |
| tlsnap | 2390719 | timeline on `worktrees/pf5mem` (pf5 + the local MEM_SNAPSHOT_DIR probe): torch allocator history, snapshots at startup and 300/600/880 s |
| dec1..dec4 | 2390686-89 | n = 1..4 requests decoding at ~5K and ~50K, nsys (`prof_load_n.py`) |
| sizes | 2390690 | pure prefill steps 512-4000 tokens, fresh and after a cached 60K, nsys |

Every arm runs `mem_sampler.py` from before the server starts: NVML per GPU
(used, per process, util, power, clock, PCIe/C2C rx/tx), `/proc/meminfo`,
per-NUMA-node meminfo (CPU nodes 0-3, HBM nodes 4/12/20/28), per vLLM
process smaps_rollup / status, `numastat -p` every 30 s.

Scripts: `launch.sh`, `launch_snap.sh` (login side), `node_run.sh` (one arm on
the node), `timeline_run.py` (replay; configurable windows, `/metrics` dumps),
`pyspy_run.sh`, `arm_snap.sh`, `mem_sampler.py`. Analysis: `mem_report.py`,
`swarm_steps.py` (steps of the nsys windows joined with request state),
`metrics_report.py` (server-side request phase histograms), plus
`../2026-10-15-c4-timeline/{reqs,pyspy_agg,step_timeline}.py`,
`../2026-10-07-decode-mem-dive/mem_snapshot.py`.

Raw data: `/e/fscratch/profound/naeimitabiei1/sliced-prof/sw-<arm>-<hold>/`.

## Results

### Run level (four 900 s timeline arms; no-profiler arm `tlmem` quoted)

All 192 turns finish in 807-869 s in every arm (agents run out of replay).
TTFT median 0.38-0.42 s; follow-up p90 1.0-4.8 s; decode median 100-104 tok/s
per request; acceptance 2.78-2.82 tokens/verify; 170-177K tokens generated;
prefix hit 94.3% of prompt tokens (41.5M prompt, 39.1M cached, 2.38M
computed). Arms agree to a few % (py-spy and nsys windows do not move the
run-level numbers; each nsys stop stalls the engine ~7 s, so take timings from
`tlmem`).

Engine time (nsys windows, 14 x 6 s): decode verify 51-53%, prefill chunks
44-47%, draft passes 2.8%. Estimate over the run: 2.38M computed tokens x
~0.6 s per 4K chunk ~ 350 s prefill; ~16.8K verify steps x ~27 ms ~ 450 s
decode.

**Slow turns are KV-pool overflow, not frontend or scheduler.** From ~440 s
the four live contexts sum to 930K-1,007K tokens against the 800K pool
(2 x 400K): append-only turns of the largest agent are evicted and recompute
(`prefix_diverge.py`: #200 -> #201 is 100% append-only, TTFT 48-50 s in all
four arms; #133 37-44 s, #195 51-61 s). Plus the deterministic client-side
ones (#161 tool list change, #162 / #154 / #155 new conversations) and the
four cold first turns (58-294K, TTFT 9-114 s). ~150 s of the ~850 s run is
overflow recompute, during which decoders also starve.

**Decode starvation during long prefills** (interleave 1): gen rate drops to
11-16 tok/s total for 30-100 s at a time (bins at 0-100, 440-470, 530-570,
640-670 s): one 27 ms verify step + 2 draft passes per ~0.6 s chunk (4.5%).

### Request path (CPU), py-spy over the whole replay (`an/pyspy-*.txt`)

| stage | cost |
|---|---|
| API main thread (uvloop: HTTP, Anthropic conversion, SSE) | 35.4 s active in ~920 s (3.8% of a core); 39% of it the detokenizer (`detokenizer.py:225`) |
| chat template (jinja) + tokenize (fastokens) | 2.5 s for 192 requests (~13 ms each) + AnyIO thread 4.8 s |
| EngineCore input thread (msgpack decode + sha256 block hashing) | 1.4 s total |
| EngineCore scheduler / output | ~2% of its samples busy; the rest spins on the workers' shm queue |
| worker 0 host | execute_model 42% of samples (prefill's eager launches: `tiered_prefill_moe`, `build_prefill_chunk_metadata` 3.6%); GPU still 97-98% busy in prefill |

Not a bottleneck anywhere on the CPU side.

### Decode step by batch size and context (nsys, ms/step, 4-rank mean)

Period = verify launch to verify launch (incl. 2 draft passes and sampling).
`an/bd-dec-table.txt` (5K / 50K), `an/bd-ctx-table.txt` (100K / 190K /
250-380K), `an/bd-tl-table.txt` (the swarm's own steps).

| n (M) | 5K | 50K | 100K | 190K | 250-380K | swarm (real) |
|---|---|---|---|---|---|---|
| 1 (4) | 16.80 | 17.67 | 16.88 | 17.91 | 17.54 (380K) | |
| 2 (8) | 19.20 | 20.61 | 20.25 | 20.73 | 20.60 (300K) | 22-24 (verify only) |
| 3 (12) | 21.89 | 24.18 | 23.01 | 23.43 | 23.61 (250K) | 24-27.5 (verify only) |
| 4 (16) | 23.11 | 26.32 | 25.83 | 26.93 | - | **27.02** |

Swarm decode step (4 decoders at 100-330K, M=16), 27.02 ms: MoE expert GEMMs
12.89 (48%; `layer_kernel` 13.0 + `finalize_kernel` 1.58 kernel-time),
decode_gemm 3.41, outside the verify graph 2.10, DCP collectives 1.60,
AR + RMSNorm 1.49, indexer 1.31, FlashMLA 1.09 + combine 0.26, small cuBLAS
1.08, route/act/router 0.56, other graph kernels 0.47, idle 0.47, KV write
0.27. Skip-layer KV staging (`_mark/_remap/_compact/_gather_rows`, UVA reads
of the Grace KV, a parallel branch of the verify graph) is ~2.0 ms/step of
kernel time but almost fully hidden (`stage_exposed.py` + a contention
check): 15 us/step with no other kernel running, no main-chain kernel ever
waits on it, and the kernels it overlaps run slower by ~0.10 ms/step (swarm
window) to 0.26 ms/step (4 x 190K), mostly decode_gemm (17.6 -> 20.7-23.8
us) and the Lamport AR. Exposed cost ~0.1-0.27 ms/step (0.4-1%).

Context costs little in decode (n=4: 25.8 at 100K -> 26.9 at 190K; the
indexer 0.84 -> 1.23); the step grows with M through the MoE (n=1 -> 4:
5.1 -> 12.9 ms) and everything else stays near-flat (decode_gemm 3.06 -> 3.41,
DCP 1.16 -> 1.60, AR 1.07 -> 1.49). Per-request decode at n=4: 72-75 tok/s
vs 75-87 at n=1.

### Prefill (nsys)

| | ms | MoE | NCCL AG | dense | attn | NCCL AR | indexer | norms | idle |
|---|---|---|---|---|---|---|---|---|---|
| 4K chunk @ ~100K (lone 300K prompt) | 582 | 246 | 79 | 79 | 62 | 43 | 55 | 33 | 2% |
| @ ~200K | 611 | 243 | 80 | 76 | 60 | 43 | 84 | 32 | 2% |
| @ ~290K | 650 | 241 | 84 | 74 | 60 | 46 | 116 | 32 | 2% |
| swarm chunks (median, 37-38 steps) | 553-618 | | | | | | | | 2-3% |

A 300K cold prompt alone: ~45 s un-profiled (73 chunks; the measured 62.6 s
includes three ~7 s nsys stops). Short prefills (`an/sizes-table.txt`,
un-profiled TTFT): 512 / 1K / 2K / 4K fresh 0.12 / 0.24-0.28 / 0.25 / 0.50 s;
after a cached 60K 0.22 / 0.31 / 0.38 / 0.71 s. The indexer is the only
component growing with context (55 -> 116 ms per chunk); NCCL all-gather +
all-reduce stay ~125-130 ms per chunk (~21%).

### VRAM (per GPU, 95.6 GiB; `an/mem-*.txt`, `an/snap-rank0-*.txt`)

Startup (s from launch): dist init +2 GiB (65 s); weights load 99-443 s
(page-cache bound, 420 GB checkpoint from GPFS; 87.6 GB used); KV +5.8 GB
(510 s); graphs +1.5 GB (562 s); ready at ~563-580 s.

Serving: 96.48 GB used (NVML), flat over the whole run (+2 MB). Probe
(`phases-rank0.jsonl`): torch allocated 88.45 GiB the whole run; reserved
89.47 at startup -> 90.96 after the first long prefills; free 2.82 -> 1.33
GiB. The +1.49 GiB is 17 new allocator segments, all free between steps: the
long-context prefill working set (tiered prefill MoE 448 MiB, DCP q
all-gathers 288 + 256 MiB, indexer 132 MiB, prefill attention 90 MiB, ...).
The startup profile run does not see it; the 3.6 GB reserve absorbs it.

Live at serving, rank 0: routed experts 70.73 GiB (14,290 hot); dense bf16
~9.2 (o_proj 3.70, qkv_a 2.47, q_b 1.23, shared expert 0.91 + 0.46, kv_b
0.54, indexer 0.34, router 0.25); KV on HBM 3.27 (22 anchor layers +
indexer); cold-prefetch slots 0.92; workspace 0.82; MTP layer MoE (Marlin)
1.27; embed 0.44; graph pool 0.53 (0.50 free). Non-torch ~3.3 GiB.

### DRAM (Grace, 4 x ~120 GiB)

Per worker (bound to its own NUMA node): 41.5 GB RSS, all anon, flat during
serving: +29 GB at the residency plan (cold experts, 4,910 x 5.1 MiB = 24.5
GiB, + pinned staging), +10 GB at KV allocation (57 skip layers' KV on Grace
~7 GiB, drafter KV, embedding 0.44). API server 3.2 GB, EngineCore 1.8 GB.
Per CPU node: 112.6 of ~120 GB used = 45 GB anon + 62 GB page cache (the
checkpoint) + 4 GB pre-existing mlock; truly free ~6 GB/node, page cache
reclaimable. MemTotal 878 GiB counts the HBM NUMA nodes (cudaMalloc'd HBM
shows as used memory of nodes 4/12/20/28).

GPU telemetry (NVML 1 Hz): SM util 100% median, ~500 W, SM clock 1.92-1.95
GHz median while serving.

## Opportunities (for the per-item follow-ups)

1. **KV pool too small for this swarm**: ~150 s of 850 s recomputing evicted
   append-only prefixes once contexts sum > 800K. Pool 3 x 400K costs ~+1.35
   GiB HBM per rank (22 of 79 layers' KV on HBM) ~ 270 hot experts, and +3.5
   GiB Grace per rank (page cache covers it). Or: evict by cost / keep the
   live agents' prefixes (LRU evicts a waiting agent's whole context).
2. **Decode starvation behind long prefills** (4.5% decode duty for 30-100 s):
   more decode steps per chunk, or smaller chunks at long context.
3. **Decode MoE at M=12-16** (48% of the swarm step; finalize_kernel still
   1.58 ms/step at M=16).
4. **Prefill at long context**: indexer 55 -> 116 ms per chunk (19% at 290K);
   NCCL AG + AR ~21% of every chunk.
5. ~~Skip-layer KV staging~~: 2 ms/step of kernel time but only ~0.1-0.27
   ms/step exposed (contention); not a lever.
6. **HBM**: 1.33 GiB stays free while serving (reserve 3.6 sized for the
   long-prefill working set); a fixed prefill workspace would let the reserve
   (hence the hot set or the KV pool) grow.
7. Not opportunities: API server, tokenizer, scheduler, IPC, block hashing
   (all < 4% of one core); GPU idle 0.47 ms/decode step, 2% of prefill.
