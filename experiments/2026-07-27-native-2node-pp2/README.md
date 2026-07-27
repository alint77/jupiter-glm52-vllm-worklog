# Native vLLM GLM-5.2 on 2 nodes, TP4 × PP2 (c1 and c4)

## Goal

Measure the **plain (non-tiered) vLLM** serving path for `GLM-5.2-W4A16` on
**two Booster nodes** with `tensor-parallel-size 4` (intra-node, NVLink) and
`pipeline-parallel-size 2` (cross-node, boundary only), at batch one (c1) and
at 4-concurrent DCP4 @ 400K (c4).

Why this is worth running: on 2 nodes TP4×PP2 the whole 361 GB model fits in
HBM across 8 ranks (~45 GB/rank), so **no Grace/CPU offload is needed** —
unlike the 1-node native baseline (37.6 tok/s, 40 GB/rank UVA offload) and
unlike the tiered single-node system. This is a clean "throw a second node at
HBM residency" baseline. It is also distinct from the 1-node TP2×PP2 config
the SOL analysis rejected: there each stage had only 2 GPUs and half the HBM
bandwidth; here each stage keeps the full 4-GPU NVLink fabric, and only the
pipeline-boundary activation crosses nodes.

Contrast points:
- c1 vs the 37.6 tok/s single-node native baseline (offload cost removed, PP
  bubble added — net direction is the measurement).
- c4 vs the tiered c4 ~185 tok/s aggregate (native, no MTP, no Grace tier).

No MTP, no tiered MoE, no `--cpu-offload-gb`. Plain W4A16 target
(`GLM-5.2-W4A16-55c92ae`), `GlmMoeDsaForCausalLM` loader, `fp8_ds_mla` KV,
EP4, CUDA graphs (Inductor mode 3).

## Configurations

Common (`run-server.sh`):

```
vllm serve $GLM52_W4A16_MODEL --served-model-name glm52-w4a16 --host 0.0.0.0 --port 8000
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --distributed-executor-backend mp
  --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER_ADDR --master-port $PORT
  --enable-expert-parallel --enable-ep-weight-filter --numa-bind
  --kv-cache-dtype fp8_ds_mla --max-model-len 400000 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90 --optimization-level 2
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE",...,"fuse_allreduce_rms":false}'
```

| | c1 (`job-c1.sh`) | c4 (`job-c4.sh`) |
| --- | --- | --- |
| `--max-num-seqs` | 1 | 4 |
| `--decode-context-parallel-size` | 1 | 4 |
| model runner | V1 (default) | V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) |
| capture / compile sizes | `[1]` | `[1,2,3,4]` |

Slurm: `--nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=288
--partition=booster --account=profound --time=04:00:00`. Each task runs one
`vllm`; node-rank 0 is the leader (API server + EngineCore + 4 workers = PP
stage 0), node-rank 1 runs 4 workers = PP stage 1. Per-job scratch caches
(`JOB_CACHE`) avoid the parallel-cache-corruption trap. `NCCL_NET_GDR_LEVEL=PHB`,
`NCCL_DEBUG=WARN`.

## Correctness gate (run before any throughput)

`run-correctness.sh <node0> <result_dir>` verifies greedy output is bit-exact
vs the single-node native baseline (`agent_space/baseline/`):

1. Semantic smoke: `"The capital of France is"` → ` Paris. Distance from Paris to Lyon is`.
2. 4K/seed-11/256 generated-text SHA-256 == `692e494f...`.
3. 400K/seed-13/256 generated-text SHA-256 == `d594e4d4...` (project golden SHA).

DCP4 and PP2 should both be lossless (Phase 15 reproduced the golden SHA under
DCP4). A mismatch flags a PP layer-split or DCP×PP numeric issue.

## Reproducing

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

# submit both 2-node jobs in parallel
sbatch agent_space/experiments/2026-07-27-native-2node-pp2/job-c1.sh
sbatch agent_space/experiments/2026-07-27-native-2node-pp2/job-c4.sh
squeue -u "$USER"

# once each job's node 0 is healthy (cold start ~20-30 min):
node0_c1=$(scontrol show hostnames $(squeue -h -n native-pp2-c1 -o "%N" | head -1) | head -1)
node0_c4=$(scontrol show hostnames $(squeue -h -n native-pp2-c4 -o "%N" | head -1) | head -1)

exp=agent_space/experiments/2026-07-27-native-2node-pp2
agent_space/experiments/2026-07-27-native-2node-pp2/run-correctness.sh "$node0_c1" "$exp/c1"
agent_space/run-batch1-baseline.sh "$node0_c1" "$exp/c1"      # 4K/32K/400K c1

agent_space/experiments/2026-07-27-native-2node-pp2/run-correctness.sh "$node0_c4" "$exp/c4"
agent_space/experiments/2026-07-27-native-2node-pp2/run-bench-c4.sh   "$node0_c4" "$exp/c4"
```

`run-batch1-baseline.sh` is the existing c1 client (warmup-4k, 4k-256, 32k-256,
399744-256), directly comparable to the 37.6 tok/s baseline.

## Risks

- **DCP4 × PP2** is untested on this stack (primary). Fallbacks: DCP2; then
  plain `--max-num-seqs 4` at reduced `--max-model-len`. c1 is independent.
- **V2 runner × PP2** untested. Fallback: V1 + PP2 (+ DCP4 if it works on V1).
- **First cross-node NCCL** may need `NCCL_SOCKET_IFNAME` tuning.
- If `mp` multi-node misbehaves, fallback to `external_launcher` (`srun
  --ntasks=8`) or Ray (2.56 installed).

## Results

Pending. Will record job IDs, c1 4K/32K/400K decode tok/s, c4 4K/400K aggregate
tok/s, correctness pass/fail, free HBM, and comparison to the 37.6 single-node
native baseline and the tiered c1/c4 numbers.
