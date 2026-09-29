# Overnight plan (written 2026-09-29 02:15, resume 03:22)

User goal: GLM-5.3 W4A16 decode step time at c=1, DFlash2 k=7, DCP4, 400K, reserve 7,
prefix caching on (serve.sh defaults). Execution/kernel work only: no quantization,
no pruning / expert-count / drafter changes, no KV offload, stay DCP4, same ops.
Benchmark: MiMo task set via bench_node.sh / agentic_bench.py (chat endpoint, T=1.0,
top_p 0.95, seeds); compare with paired.py (same requests, fit with arm offset).
Nodes: `sbatch --time=02:00:00 agent_space/experiments/2026-09-27-glm53-mtp7-profile/hold.sbatch`
(full node, gpu:4), run with onnode.sh (HOLD_JOB=...) or seq_arms.sh; launch with
`setsid nohup` (never plain `&`); never edit vllm/ while a server is about to import it.
Caches on /e/fscratch (VLLM_CACHE_ROOT, TRITON_CACHE_DIR). Commit vLLM locally only
(source agent_space/jupiter-env.sh first; trailers Co-Authored-By + Claude-Session +
Signed-off-by: alint77 <ali.nt1377@gmail.com>); never push.

## State at 02:15
- Committed a98ca6938c: fused one-shot DCP ops (bitwise identical, -0.46 ms/step
  in isolation). Task-set A/B running: fused on 2111470 (rows-dcpfused-c) and
  2111454 second arm (rows-dcpfused-b); unfused rows-dcpunf-b (2111454),
  rows-dcpunf-a (2111453). Interim (29 reqs): -0.29 ms/step, CI -0.62..+0.05.
  Analyse: `paired.py --a rows-dcpunf-{a,b}.jsonl --b rows-dcpfused-{b,c}.jsonl`.
- node_a2.sh queued on 2111453: eager (CUDAGRAPH_MODE=NONE, torch.compile on)
  stack-traced profile -> traces/glm53-agentic-eager-stack-2111453 for glue
  attribution (per-layer sequence in dive/r0-w0-layers.txt).
- Holds 2111453/2111454 end ~03:31, 2111470 ~02:39: resubmit as needed.

## Real step (nsys, graph-level): 25.56 ms = graph 23.55 + other GPU 1.71 + idle 0.31.
Outside graph: verify lm_head 140 us, NCCL logits all-gather 24 us, _topk_topp 171-202 us,
DFlash2 fc 137 us (replicated 453 MB, at floor), drafter 6 layers ~0.94 ms,
drafter lm_head 139 us, input prep ~0.1 ms. Host is never late.

## Queue (in order)
1. Glue in sparse-MLA/DCP path (exact): apply upstream #50365 (single-tile index
   remap, no torch.zeros counter/atomics; `git apply` of `gh pr diff 50365` checks
   clean) + #57458 idea (single-tile compaction writes the -1 tail -> drop
   torch.full_like(-1) in triton_filter_and_convert_dcp_index). Test equality vs old
   kernel (valid prefix as a set / count) at [8, 2048]. Then attribute the rest from
   the eager trace (#16 int->bool AUnary, #17 memcpy32_post, #18 elementwise,
   #30/#31 in MoE path) and remove what is redundant.
2. Reuse converted indices on layers that skip their own top-k (upstream #53562 /
   #49678 idea; only ~21 of 78 layers run the indexer top-k): skip filter/convert on
   the other ~57 layers.
3. #58985 idea: FlashMLA sparse fp8 decode walks all 2048 slots though a DCP4 rank
   owns ~512 (compacted to front) - measure per-layer sparse kernel time vs a
   topk_length-aware variant; needs FlashMLA kernel patch (vllm-project/FlashMLA#27).
4. AR + residual + RMSNorm fusion (~0.5 ms): first probe NVLink multicast on the node
   (upstream #48075: FlashInfer mnnvl workspace created without multicast then
   faults); try VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm with FUSE_AR_RMS=true;
   else own kernel on custom-AR buffers. Rounding-equivalent, not bitwise: check
   acceptance + GSM8K.
5. Sampler _topk_topp_kernel (grid 8x1, ~0.17-0.2 ms): exact faster variant.
6. Small dense GEMMs / split-K reduce (~0.5 ms), drafter GEMMs, NCCL logits
   all-gathers -> one-shot.
Every change: unit test for equality, then task-set A/B (two nodes per arm, paired.py).
Log results in README.md here; keep HANDOFF.md current.
