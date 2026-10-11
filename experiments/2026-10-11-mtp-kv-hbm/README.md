# MTP layer KV in HBM under skip_host_uva

2026-10-11. Asked: put the MTP layer's MoE experts all in HBM with the planner
aware, and make its attention use the Phase 1 kernel. Checked first:

- **MoE**: already untiered (`default_loader.py` returns early for the
  deepseek_mtp drafter under tp_sliced) and charged by the planner as
  EP_SHARDED non-routed bytes. No change.
- **Attention**: the drafter already runs the Phase 1 schedule and kernel.
- **Bug found**: the MTP layer 78's MLA KV was placed on Grace (host_uva).
  `plan_glm_kv_cache` charged it to HBM as an indexer layer, but
  `get_tiered_kv_memory_tier` used the 78-layer anchor list. Fixed in vLLM
  `c047194130` (`_glm_anchor_layers`, which adds the MTP layers). Test:
  `test_skip_host_uva_keeps_the_mtp_layers_kv_in_hbm_as_planned`.

## Kernel level (nsys, `mkvsol-c4/c16` logs)

| | c16 base | c16 new | c4 base | c4 new |
| --- | ---: | ---: | ---: | ---: |
| drafter FlashMLA, 64/16-token call (us) | 137.8 / 39.9 | 24.7 / 14.2 | 38.2 / 15.2 | 13.7 / 10.2 |
| time outside the graph (ms/step) | 3.31 | 3.14 | 2.28 | 2.23 |

## Served A/B (`ab_launch.sh`, `ab_table.py`; base fmla-p1, new mtpkv)

| context | requests | base ms | new ms | per node (new - base) | mean |
| --- | ---: | ---: | ---: | --- | ---: |
| 5K | 4 | 27.69 | 27.33 | -0.28 -0.43 | **-0.36** (-1.3%) |
| 50K | 4 | 28.44 | 28.09 | -0.51 -0.19 | **-0.35** (-1.2%) |
| 5K | 8 | 38.43 | 38.30 | +0.03 -0.30 | **-0.14** (-0.4%) |
| 50K | 8 | 40.54 | 40.61 | +0.27 -0.14 | **+0.06** (+0.2%) |
| 5K | 16 | 55.57 | 55.91 | +0.15 +0.54 | **+0.34** (+0.6%) |
| 50K | 16 | 58.17 | 58.40 | +0.39 +0.05 | **+0.22** (+0.4%) |

The kernel win (~0.17 ms/step at c16) is real but within served noise. The
fix is a correctness fix (the placement now matches the plan). Startup free
HBM at c16 is 2.61 GiB (was 2.86); c4 2.84. Live servers pick it up on restart.
