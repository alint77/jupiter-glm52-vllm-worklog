# GLM-5.2 FP8 MTP graft

The official `zai-org/GLM-5.2-FP8` layer 78 was grafted onto the pinned
W4A16 checkpoint so vLLM can use the model's native MTP head while retaining
the tiered W4 target model.

## Checkpoint

- Output: `models/GLM-5.2-W4A16-FP8-MTP`
- Base shards: eight hard links to the existing 361 GiB W4 checkpoint
- MTP delta: 1,569 tensors and 10,032,632,960 payload bytes in three shards
- Mixed formats: W4A16 target layers 0-77 and FP8 block-128 MTP layer 78

The delta came from `dnhkng/GLM-5.2-AWQ-INT4-FP8-MTP-delta`. Its graft
script was corrected to calculate safetensors payload bytes rather than file
sizes (which include headers); the resulting index reconciles exactly at
397,699,787,648 bytes.

## Runtime changes

The minimal vLLM changes select `mtp_quantization_config` only for the MTP
decoder layer, retain the target's AWQ configuration elsewhere, and give the
draft layer its native EP4 ownership. The target loader drops layer-78 experts;
the draft loader skips all backbone experts before payload reads and loads only
64 of 256 FP8 experts per rank.

The physical planner accounts for 2,643,873,344 additional persistent bytes
per rank and one extra main/indexer cache pair (315,250,432 bytes at 400K).
The resulting 10 GB-reserve plan uses 2,870 hot and 1,930 Grace-resident target
experts per rank. The checked-in heat profile preserves the prior trace-based
ownership optimization at that capacity.

The launch uses MTP3, TP4/EP4, strict NUMA binding, FP8 MLA cache, Inductor
mode 3, size-4 full/piecewise CUDA graphs, and FlashInfer FP8 MoE. FlashInfer
and TensorRT-LLM JIT state is redirected to scratch, and runtime compilation is
capped at four jobs.

## Validation

Ruff, bytecode compilation, JSON and shell validation, and 41 focused tests
pass. The server completed target and draft warmup, captured both CUDA graph
modes, and served the exact 400K case.

The deterministic smoke response and all eight token strings exactly match the
non-MTP baseline: ` Paris. Distance from Paris to Lyon is`.

| Case | TTFT | TPOT | Decode | Acceptance | Tokens/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 + 256 | 0.848 s | 9.669 ms | 103.42 tok/s | 78.51% | 3.36 |
| 399,744 + 256 | 114.999 s | 9.245 ms | 108.17 tok/s | 60.74% | 2.82 |

Against Phase 8, the exact-400K result improves decode from 55.16 to 108.17
tok/s (96.1%) while TTFT rises 5.8%. The 4K result improves from 51.74 to
103.42 tok/s. Per-position acceptance is 88.16/77.63/69.74% at 4K and
92.22/57.78/32.22% at 400K.

Idle free HBM after the smoke request was 9,457-9,470 MiB per GPU. After the
fully populated exact-400K request it was 3,311-3,324 MiB; the maximum-length
request completed without an allocation failure.

## Startup notes

The target GPFS read took 912 seconds while the filtered MTP draft took 18
seconds. The first FlashInfer FP8 MoE load required a 180-object build; doing
that once outside the loaded model with memory interleaved across the four
Grace domains avoided NUMA-local compiler OOM. `MAX_JOBS=4`,
`FLASHINFER_WORKSPACE_BASE`, and `TRTLLM_DG_CACHE_DIR` keep subsequent JIT work
bounded and off the small home filesystem. MTP3 also requires graph capture
and compile sizes divisible by four.
