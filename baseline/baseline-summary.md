# GLM-5.2 W4A16 CPU-offload baseline

Date: 2026-07-16

## System

- JUPITER Booster node: `jpbo-021-29`
- Slurm allocation: `954640`
- vLLM commit: `d08eebad162bbd1f2e99cca550313daaa81c7654`
- Model: `GLM-5.2-W4A16-55c92ae`
- Tensor parallel: 4
- Expert parallel: enabled, 64 of 256 experts per rank
- UVA expert offload: 40.33 GiB per rank
- Model memory: 54.82 GiB per rank

## Serving configuration

- `--offload-backend uva`
- `--cpu-offload-gb 40`
- `--cpu-offload-params experts`
- `--kv-cache-dtype fp8_ds_mla`
- `--max-model-len 400000`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.90`
- `--optimization-level 2`
- Inductor mode 3 with max autotune and coordinate-descent tuning
- Full and piecewise CUDA graphs, capture size 1
- `fuse_allreduce_rms=false` because the fused path failed warmup on this stack
- Per-rank NUMA binding to Grace CPU nodes 0, 1, 2, and 3

## Capacity and idle memory

- Available KV cache memory: 28.85 GiB per rank
- GPU KV cache capacity: 574,336 tokens
- Maximum concurrency at 400,000 tokens: 1.44x
- Idle HBM: approximately 90.7 GiB used and 6.58 GiB free per GPU
- CUDA graph pool: 0.06 GiB per rank

## Batch-1 results

All requests used random input, deterministic decoding, `ignore_eos`, and 256
generated tokens.

| Input tokens | TTFT | Approx. input/TTFT | TPOT | Steady decode | Total duration |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.990 s | 4,139 tok/s | 26.98 ms | 37.06 tok/s | 7.87 s |
| 32,768 | 7.652 s | 4,282 tok/s | 26.99 ms | 37.06 tok/s | 14.53 s |
| 399,744 | 120.853 s | 3,308 tok/s | 26.62 ms | 37.57 tok/s | 127.64 s |

The input/TTFT column is a useful end-to-end approximation, not a kernel-only
prefill measurement.

## HBM headroom finding

At `--gpu-memory-utilization 0.94`, idle HBM had only about 2.95 GiB free per
GPU. A 32K request failed in FlashMLA's FP8 sparse mixed-batch path while
allocating the 8K prefill chunk output. The stable-ABI wrapper surfaced this as
an `aten::new_empty` dispatcher failure.

At `--gpu-memory-utilization 0.90`, idle free HBM increased to about 6.58 GiB.
The same 32K request and a 399,744-token request both completed, while KV
capacity remained safely above the configured 400K context length. This is the
baseline setting.

## Result files

- `batch1-4k-256out-hbm90-warm.json`
- `batch1-32k-256out-hbm90-warm.json`
- `batch1-399744-256out-hbm90.json`
- `idle-memory-hbm90.txt`
- `server.out`
- `server.err`

