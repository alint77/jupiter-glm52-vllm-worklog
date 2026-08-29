# GLM-5.3 AutoRound W4A16 G64

Reproduces the recipe of the HF GLM-5.2 quant
(`GLM-5.2-AutoRound-W4G64-MTP-e1ba887`) on GLM-5.3, which has no AutoRound
release. The 5.2 checkpoint is the tiered-MoE production target, so a matching
5.3 quant lets the 5.2/5.3 comparison hold the quantizer fixed.

## Source

`zai-org/GLM-5.3-BF16` -- 753,329,921,024 BF16 parameters plus 19,456 F32, a
byte-identical dtype profile to `zai-org/GLM-5.2`. Published 2026-08-28.

The main `zai-org/GLM-5.3` repo is FP8 (751.2B `F8_E4M3`, only 2.1B BF16), so
quantizing from it would have been quantize-on-quantized with the FP8 release
as the quality ceiling. The BF16 repo removes that confound entirely.

## Recipe

Every value below is recorded in the 5.2 checkpoint's `quantization_config`:

| parameter | value |
| --- | --- |
| `autoround_version` | 0.14.0 (pinned) |
| `bits` / `group_size` | 4 / 64 |
| `sym` / `data_type` | True / int |
| `batch_size` | 2 |
| `gradient_accumulate_steps` | 4 |
| `nsamples` | 512 |
| `low_gpu_mem_usage` | True |
| `packing_format` | `auto_round:auto_gptq` |

`iters`, `lr`, `minmax_lr`, `seqlen` and `dataset` are **not** recorded, and are
deliberately left unset. auto-round is pinned to 0.14.0 -- the version that
produced the 5.2 quant -- so its own defaults fill them in identically.
Substituting guesses would diverge from the recipe rather than match it.

## What gets quantized

Confirmed against the 5.2 weight index rather than inferred from the config.
Only routed expert MLPs carry `qweight`/`qzeros`/`scales`:

| module | 5.2 treatment |
| --- | --- |
| `layers.{3..78}.mlp.experts.*.{gate,up,down}_proj` | **W4 G64** |
| `layers.*.mlp.gate` (router) | BF16 |
| `layers.*.mlp.shared_experts.*` | BF16 |
| `layers.*.self_attn.*` incl. `indexer` | BF16 |
| `layers.{0,1,2}.*` (dense) | BF16 |
| `eh_proj`, `weights_proj` | BF16 |

Layer 78 is the MTP block and **is** quantized. That is a concrete win over the
NVFP4 checkpoint, which leaves it BF16 at 4.50 GiB per rank against W4G64's
1.21 -- 3.29 GiB per rank the placement profile currently has to work around.

`layer-config.json` carries the ten generic BF16 regexes from the 5.2
`extra_config` plus `model.layers.78.mlp.gate`, the one explicit entry no
regex covers. The other 862 entries in that config are AutoRound expanding
those regexes at save time, not recipe information.

## Layout

- `download-bf16.sh` -- pulls the 1.5 TB BF16 source to fscratch, resumable
- `layer-config.json` -- the BF16 keep-list
- `quantize.sbatch` -- the Booster job
- `glm52-recipe.json` -- the reference `quantization_config`, for diffing

## Status

BF16 download in progress. Quantization not yet launched.
