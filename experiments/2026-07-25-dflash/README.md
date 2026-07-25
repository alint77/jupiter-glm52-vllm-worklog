# DFlash speculator for GLM-5.2

`UCloud-org/GLM-5.2-FP8-DFlash` — 7.0 GB, **5 layers**, hidden 6144, 64 heads,
`num_target_layers: 78`, `target_hidden_size: 6144`, vocab 154880 (no remap).
Native DFlash checkpoint (`architectures: ["DFlashDraftModel"]` → registry entry
`qwen3_dflash.DFlashQwen3ForCausalLM`), not speculators-format.

Two properties that decide the outcome:

- **`block_size: 16`** → verify batch **16-17**, against DSpark's 9 and MTP3's 4
- **`layer_types` all `full_attention`** (not SWA), so
  `dflash_has_any_non_causal()` is True and the draft needs a non-causal-capable
  backend — set via `speculative_config.attention_backend: FLASH_ATTN`

Its `dflash_config` is a **nested** dict carrying `target_layer_ids` and
`mask_token_id`, which is the layout `qwen3_dflash.py` was written for. The
top-level-attribute bug that cost DSpark blocker 5 cannot recur here.

## Result: loads first try, and is the slowest of the three

Job 1043233, c1, DCP1, V2 runner, `num_speculative_tokens=15` (DFlash uses the
`1+N` bonus-anchor layout, so 15 proposed from a block of 16). Correct
deterministic completion. **No integration blockers at all** — the seven cleared
for DSpark, plus upstream #48776, left the path clean.

| | MTP3 (V2) | DSpark t=8 | **DFlash t=15** |
| --- | ---: | ---: | ---: |
| verify batch | 4 | 9 | **16** |
| acceptance length | ~2.9 / 4 | 3.98 / 8 | **6.84 / 15** |
| realistic mean TPOT | — | 10.65 ms | **14.19 ms** |
| implied decode tok/s | **106.08** | ~93 | **~70** |
| realistic end-to-end | **95.44** | 84.67 | **65.57 / 65.99** |
| implied step time | ~27 ms | ~42 ms | **~97 ms** |

**DFlash has by far the best acceptance and by far the worst throughput.** That
is the cleanest possible demonstration that acceptance length is the wrong
figure of merit: what matters is accepted tokens per unit time, and the three
land in strict inverse order of verify-batch width.

Step time against verify width — 27 ms at 4, 42 ms at 9, 97 ms at 16 — is
superlinear here, steeper than the 1.06 ms/token routed-MoE slope measured at
c4, because the draft models themselves also differ (DFlash is 5 full-attention
layers, DSpark 3 sliding-window layers).

## Verdict

Not viable for this target, and for a structural reason: a W4A16 tiered-MoE
target whose verify cost grows with batch size penalises wide speculative
blocks. The ranking MTP3 > DSpark > DFlash follows block width exactly, and
matches the prediction made before the run.

This also independently confirms the HANDOFF's earlier decision to deprioritise
DFlash, though for a different reason than the one recorded there (that was
about ~3K-context training versus 400K acceptance; this is about verify-batch
economics, visible already at short context).
