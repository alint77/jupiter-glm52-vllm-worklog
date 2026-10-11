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

Note: Parastation runs sbatch scripts from a copy under
`/var/spool/parastation/jobs/<id>`, so `BASH_SOURCE`-relative repo
detection silently resolves wrong and relative writes fail (job 2287088
died in 54 s on exactly this). All scripts here carry the
`claude-server.sbatch` hardcoded-fallback repo detection.

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

Status: complete for tonight — standby job 2287122 (jpbo-017-41). probe1 on
GPUs 0 and 2, probe2 cold/hot sweep on GPUs 1 and 3.

Platform facts (from `logs/probe1-inventory-*.txt`):

- NVIDIA GH200 120GB, driver 595.71.05, CUDA 13.2; 97,871 MiB visible per GPU.
- Exactly **one memory clock bin: 2619 MHz** (max = current), constant
  through the runs — there is nothing clock-side to vary on this platform.
- ECC mode **Enabled/Enabled** (current/pending) on all four GPUs.
- Topo: GPU k owns CPUs [72k .. 72k+71] and NUMA node k; a separate
  "GPU NUMA ID" column (4/12/20/28) marks the memory-only HBM nodes.

probe1, GPU 0, NUMA-bound to node 0, bf16, 8 GiB span (GiB/s, ratios vs
stream copy):

| pattern            | GiB/s   | x copy |
|--------------------|---------|--------|
| copy_6g_rw (R+W)   | 3395.8  | 1.000  |
| fill_6g_w (W only) | 3156.3  | 0.929  |
| sum_bf16_6g_r (R)  | 3300.1  | 0.972  |
| gather 2 B random  | 122.9   | 0.036  |
| gather 256 B random| 791.3   | 0.233  |
| gather 64 KiB random | 3084.7 | 0.908 |

GPU 2 replicate agrees within 0.3% on every pattern (3397.5 / 3154.9 /
3298.3 / 122.9 / 791.0 / 3075.3); the platform is homogeneous and
single-GPU microbenches are representative.

probe2 sweep (GPU 1; GPU 3 identical to within 1%), random rows from an
8 GiB span, hot = indices re-drawn from a fixed 4096-row set:

| granularity | cold GiB/s | hot GiB/s | hot/cold |
|-------------|-----------:|----------:|---------:|
| 2 B         | 122.8     | 290.2     | 2.36     |
| 4 B         | 241.7     | 546.0     | 2.26     |
| 8 B         | 459.8     | 934.3     | 2.03     |
| 32 B        | 99.2      | 99.2      | 1.00     |
| 128 B       | 396.2     | 396.2     | 1.00     |
| 256 B       | 790.8     | 791.2     | 1.00     |
| 1 KiB       | 3092.0    | 3126.5    | 1.01     |
| 4 KiB       | 3081.6    | 4485.7    | 1.46     |
| 16 KiB      | 3052.8    | 4301.0    | 1.41     |
| 64 KiB      | 3099.0    | 4020.8    | 1.30     |

Readings:

- Stream copy hits ~3.4 TiB/s (3.65 TB/s decimal), roughly 0.9x of the
  4 TB/s-class expectation for this HBM3 config, so the streaming
  infrastructure is healthy; these numbers are trustworthy references.
- Fine grain (2-8 B): measured 123 / 242 / 460 GiB/s = 0.58-0.60 x the
  sector-efficiency bound (stream read x gran/32 B = 206 / 412 / 825).
  The residual ~1.7x tax beyond sector waste is the activation/scheduling
  regime that DRAM timing parameters act on — the real signal of this
  thread, and (per the hot caveat below) the only clean one.
- The 32 B and 128 B points are **kernel artifacts, not DRAM physics**:
  cold 32 B cannot be slower than cold 2 B, and hot exactly equals cold
  there (99.2/99.2, 396.2/396.3) — an issue-bound `index_select` path
  with width-dependent dispatch. 32-256 B needs a custom gather kernel
  before any DRAM-level claim; treat those three points as lower bounds.
- **>= 1 KiB random gathers run at ~3090 GiB/s = stream copy.** With >=
  1 KiB contiguous per access, HBM3 absorbs uniformly random row order at
  full rate: the gather tax is a sub-1 KiB phenomenon.
- Hot-arm caveat, correcting probe2's own design intent: the hot working
  set is 8 KB-256 MiB against a 60 MB L2, so hot/cold gaps (2.0-2.4x at
  2-8 B, 1.3-1.46x at >= 4 KiB) measure cache-vs-DRAM, not row-buffer
  state. The row-activation claim rests on the sector-bound deviation
  above, not on the hot arm.
- Pattern-level serving consequence: sub-1 KiB gathers pay the 0.04-0.24x
  tax; >= 1 KiB blocked reads (paged KV, contiguous expert slices) do not.
  Whether real serving traffic sits in the taxed band is a trace-driven
  question — do not treat this as headroom until then.

Raw JSONs in `logs/` (untracked); the durable numbers live in this table.

## Next

- Custom (triton) gather kernel to de-confound the 32-256 B band.
- Trace-driven histogram of real serving gather sizes mapped onto this
  curve, before claiming any serving-side headroom.

## Reproducing

From the repo root on a Booster login node, modules are loaded by the probe
itself (`jupiter-env.sh`); the standby needs nothing. Do not run the
bandwidth probes on the login node.
