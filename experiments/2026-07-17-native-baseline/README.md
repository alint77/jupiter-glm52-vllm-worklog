# Native UVA baseline reproduction

Clean reproduction on Slurm job `956247`, node `jpbo-027-32`, before loading
the new pageable-Grace extension.

## Configuration

- vLLM commit `d08eebad162bbd1f2e99cca550313daaa81c7654`
- TP4/EP4 with EP weight filtering and NUMA binding
- Existing UVA expert offload, 40 GiB/rank
- `fp8_ds_mla` KV cache and 400,000-token maximum model length
- 90% HBM utilization, Inductor mode 3, full/piecewise CUDA graphs
- Batch one, random deterministic prompts, 256 generated tokens

## Results

| Input | TTFT | TPOT | Decode |
|---:|---:|---:|---:|
| 4,096 | 0.978 s | 27.076 ms | 36.93 tok/s |
| 32,768 | 7.578 s | 26.886 ms | 37.19 tok/s |
| 399,744 | 118.936 s | 26.642 ms | 37.53 tok/s |

The result matches the previous baseline. Idle allocation was 90,679–90,688
MiB per GPU, leaving 6,593–6,602 MiB free.

Cold checkpoint loading from ExaSTORE/GPFS took 1,207.21 seconds. Total model
loading took 1,226.94 seconds, followed by 64.00 seconds of compilation and
about 10 seconds of profiling/graph warmup. The eight checkpoint shards read at
roughly 2.5 minutes each.

Raw server output, benchmark output, GPU memory capture, and detailed vLLM JSON
results are stored beside this report.
