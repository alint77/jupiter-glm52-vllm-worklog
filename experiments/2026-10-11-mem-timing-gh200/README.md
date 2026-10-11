# GH200 memory-timing sensitivity (research thread opener)

## Goal

Quantify how much DRAM-level memory timing state (row activation/precharge,
CAS latency, refresh) matters to AI serving access patterns — the GH200-side
half of a research thread whose other half (modifying timings on an owned
RTX 2080 Ti via BAR0 MC register RE) happens off-cluster. GH200 is a
**measurement platform only**: HBM3e on JUPITER has no user-accessible
timing path (link/timing training happens at power-on inside signed
firmware, GSP owns the memory controller, we have no host root, and
register experiments on shared Booster nodes would be node-fatal and
against site policy). Everything here is read-only.

The physiology question: streaming GEMV weight reads should be nearly
insensitive to timing (sequential row hits, prefetch); random gathers
(KV-cache, MoE expert reads) should carry a row-miss tax that granularity
controls. This is the mining-era ethash intuition (straps bought +30-40%
because DAG random reads are latency-bound) transplanted to LLM serving.

## Mechanism

A 1h full-node standby allocation holds the node so probes can be launched
into it without new sbatch round trips:

```bash
# submit the standby (jobid lands in alloc.jobid; node in alloc.node)
sbatch --parsable agent_space/experiments/2026-10-11-mem-timing-gh200/standby.sbatch

# run a script on one GPU inside it (blocks; logs to logs/<name>-gpu<k>-<ts>.log)
bash agent_space/experiments/2026-10-11-mem-timing-gh200/node-submit.sh 0 \
  agent_space/experiments/2026-10-11-mem-timing-gh200/probe1.sh
```

`node-submit.sh` reuses the verified booster pattern
(`srun --jobid ... --overlap --cpu-bind=none --gres=gpu:4`) and pins the GPU
via `CUDA_VISIBLE_DEVICES`; concurrent steps must pin different indices.

## probe1

Read-only, single GPU, NUMA-bound to the GPU's local Grace node:

- Inventory: `nvidia-smi` full, `-L`, `topo -m`, clock/ECC queries,
  `SUPPORTED_CLOCKS`, and the full `-q -d ECC` report (what the platform
  exposes read-only; ECC mode current/pending is the only memory-related
  state visible to us).
- Access-pattern bandwidth (torch, bf16):
  `copy_6g_rw` (stream R+W), `fill_6g_w` (pure W), `sum_bf16_6g_r`
  (pure R), and random `index_select` gathers at 2 B, 256 B, 64 KiB
  granularity over an 8 GiB span, reported as GiB/s and as ratios against
  stream copy.

Intended reading: the 2 B / stream ratio is the row-miss tax at worst-case
granularity; the granular sweep shows where the curve inflects. That curve
bounds what DRAM timing parameters could matter per access pattern class,
which is the input to the 2080 Ti actuator campaign.

## Results

Status: standby job 2287088, probe1 in flight. Tables filled when the hop
completes. Raw logs in `logs/` (untracked), JSON at `logs/probe1-bw-*.json`
(untracked); numbers land here on commit.

## Reproducing

From the repo root on a Booster login node, modules are loaded by the probe
itself (`jupiter-env.sh`); the standby needs nothing. Do not run the
bandwidth probes on the login node.
