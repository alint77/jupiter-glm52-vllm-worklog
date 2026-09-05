# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Price the cold-staging budget against the real checkpoint, off-node.

The plan-only CLI cannot do this: `build_tiered_moe_plan` references an
undefined `vllm_config` on the path that prices a scenario (a pre-existing
fault, unrelated to the prefetch). This calls the scenario planner directly so
the fixed point can be checked before an allocation is spent on it.
"""

import os
import sys

import vllm.envs as envs
from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    build_glm_non_routed_runtime_inventory,
)
from vllm.model_executor.model_loader.tiered_moe_physical import (
    plan_tiered_moe_scenario,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    load_tiered_moe_placement_profile,
)
from vllm.model_executor.model_loader.tiered_moe_runtime import (
    plan_tiered_glm_runtime_buffers,
)

MODEL = f"/e/fscratch/profound/{os.environ.get('USER', 'naeimitabiei1')}/models/GLM-5.3-W4A16"
PROFILE = "agent_space/profiles/glm53-w4a16-2496.json"
EP = 4


def price(min_tokens: int) -> dict:
    envs.VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS = min_tokens
    manifest = build_glm_w4a16_manifest(MODEL)
    non_routed = build_glm_non_routed_runtime_inventory(manifest, EP)
    buffers = plan_tiered_glm_runtime_buffers(
        manifest, max_num_batched_tokens=8192, ep_size=EP, sm_count=132
    )
    profile = load_tiered_moe_placement_profile(PROFILE, manifest, EP)
    scenario = plan_tiered_moe_scenario(
        MODEL,
        manifest=manifest,
        non_routed=non_routed,
        runtime_buffers=buffers,
        ep_size=EP,
        hbm_capacity_bytes=95_000_000_000,
        hbm_reserve_bytes=10_000_000_000,
        host_capacity_bytes=120_000_000_000,
        host_reserve_bytes=8_000_000_000,
        max_model_len=400_000,
        kv_block_size=64,
        kv_cache_dtype="fp8_ds_mla",
        cache_tier="hbm",
        base_hbm_allocations={},
        base_host_allocations={},
        expert_placement="linear",
        num_mtp_layers=1,
        placement_profile=profile,
        replica_assignment="off",
        dcp_world_size=1,
        max_num_seqs=1,
    )
    plan = scenario.rank_plans[0]
    fixed = dict(plan.fixed_hbm_allocations)
    largest = max(
        len(layer.cold_expert_ids) + len(layer.replica_expert_ids)
        for layer in plan.layer_placements
    )
    return {
        "hot": plan.hot_expert_slots,
        "cold": plan.cold_expert_slots,
        "staging": fixed.get("cold_staging", 0),
        "largest_cold_experts": largest,
        "expert_bytes": plan.expert_bytes,
    }


def main() -> int:
    off = price(0)
    on = price(512)
    print(f"{'':22} {'prefetch OFF':>14} {'prefetch ON':>14}")
    print(f"{'hot experts':22} {off['hot']:>14} {on['hot']:>14}")
    print(f"{'cold experts':22} {off['cold']:>14} {on['cold']:>14}")
    print(f"{'largest cold tier':22} {off['largest_cold_experts']:>14} "
          f"{on['largest_cold_experts']:>14}")
    print(f"{'cold_staging MiB':22} {off['staging'] / 2**20:>14.0f} "
          f"{on['staging'] / 2**20:>14.0f}")
    print(f"\nexpert bytes: {on['expert_bytes']:,}")
    print(f"hot experts given up for the slot: {off['hot'] - on['hot']}")
    if off["staging"] != 0:
        print("FAIL: budgeted a slot with the prefetch off")
        return 1
    need = on["largest_cold_experts"] * on["expert_bytes"]
    if on["staging"] < need:
        print(f"FAIL: budget {on['staging']} < requirement {need}")
        return 1
    print(f"OK: budget {on['staging']:,} covers requirement {need:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
