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

**The MTP head (layer 78) is deliberately left BF16 here**, unlike the 5.2
checkpoint which quantizes it. This is the one intentional departure from the
recipe. It is also moot mechanically: transformers builds
`range(num_hidden_layers)` = layers 0..77 and never reads
`num_nextn_predict_layers`, so `model.layers.78` and `eh_proj` are absent from
the model graph entirely (confirmed by meta-device instantiation, not inferred).
5.2's own `block_name_to_quantize` lists only layers 3..77, so whatever produced
its layer-78 weights ran outside the block-tuning loop -- which is also why
`model.layers.78.mlp.gate` is the single explicit BF16 entry no regex covers.

`layer-config.json` carries the ten generic BF16 regexes from the 5.2
`extra_config` plus `model.layers.78.mlp.gate`, the one explicit entry no
regex covers. The other 862 entries in that config are AutoRound expanding
those regexes at save time, not recipe information.

## Layout

- `download-bf16.sh` -- pulls the 1.5 TB BF16 source to fscratch, resumable
- `layer-config.json` -- the BF16 keep-list
- `quantize.sbatch` -- the Booster job
- `glm52-recipe.json` -- the reference `quantization_config`, for diffing

## Pre-flight

- Download validated against the hub manifest: 282/282 shards, 59,585 tensors,
  zero size mismatches, 1.507 TB.
- `glm_moe_dsa` is native to transformers 5.12.1; `auto_map` is null and the
  repo ships no `.py`, so no remote code is involved.
- `NeelNanda/pile-10k` (auto-round's default calibration set) is pre-cached
  under `HF_HOME`. `jupiter-env.sh` exports `HF_HUB_OFFLINE=1`, so an
  un-cached fetch would fail in the first minute on the compute node.
- The job unsets `VIRTUAL_ENV`/`PYTHONPATH` after sourcing `jupiter-env.sh`,
  which activates the vLLM venv and would otherwise shadow the isolated one.

## Status

Stage 1 submitted as job `1534301`, 12 h (the QOS ceiling; 24 h and 48 h are
both rejected with `QOSMaxWallDurationPerJobLimit`).

auto-round has no native resume, so if 75 blocks do not fit in 12 h the
fallback is chunking by block range: `--to_quant_block_names` scopes cleanly and
per-run outputs cover disjoint tensors, so they can be merged. Extrapolate from
the first blocks rather than discovering the wall at hour 12.
