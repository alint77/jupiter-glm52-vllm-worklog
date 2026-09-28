# GLM-5.3 routing capture on MiMo's workload, and the profile built from it

Per the decision to stop profiling GLM from the live Claude-Code snapshot and
use the same data as MiMo: the same 16 agentic tasks
(`../2026-09-26-mimo-routing-profile/tasks-{0..3}.json`), the same driver
(`../2026-08-29-glm53-routing-capture/run_agentic_capture.py`, 4 passes,
max_tokens 8192), 8-token verify steps (MTP7, as MiMo's DFlash k=7).

`capture.sbatch`: `../2026-09-27-glm53-mtp7-profile/serve.sh` with routed-expert
return, which needs the V1 runner and no DCP, so DCP1 / V1 / replicas and the
new decode kernels off -- none of which changes which experts are routed.
Four shards (jobs 2098311 (on a hold node), 2098967, 2098968, 2098969):
214 + 115 + 173 + 155 = **657 requests**, against MiMo's 609 traces.

`build_profile.sh`: merge, `traces_to_manifest.py --split-by domain` (task
families held out), `optimize_routing_profile.py --residency-mode frequency` at
**3239 hot slots per GPU** (the largest DCP4 runtime count, so the planner only
demotes), owners kept from the served profile; `finish_profile.py` then orders
each layer's list by training frequency (demotion drops the least-used first)
and adds replicas with MiMo's `mimo_replicas.py` at a budget of 2000 per GPU.
The builder places one copy per expert and only where it lowers the min-max
cold load, so it placed 1962 / 1467 / 1351 / 1462.

`profiles/glm53-w4a16-agentic-3239-r2000.json`. Held out (26,483 steps; train
43,718):

| per 8-token step | served profile (2496 listed) | new profile |
|---|---|---|
| mean cold experts per GPU | 214.9 | **114.0** |
| busiest-GPU cold experts | 262.6 | **152.4** |

(The served profile is promoted to ~3211 hot at runtime under DCP4, in
expert-id order; this table is at its listed 2496.) Served A/B: `ab-gA`,
`ab-gB` in `../2026-09-27-glm53-mtp7-profile`.
