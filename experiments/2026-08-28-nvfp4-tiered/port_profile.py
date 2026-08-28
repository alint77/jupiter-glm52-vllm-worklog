#!/usr/bin/env python3
"""Port a placement profile onto the NVFP4 target as a bring-up placeholder.

**The ranking in the result is wrong and is meant to be.** It is GLM-5.2's
hot-expert ranking, and GLM-5.3's post-training moved the router, so which
experts are hot has changed. Placement does not affect whether the model loads
or what it outputs, so this is fine for bring-up and useless for performance.
Re-derive from a routing capture on the NVFP4 target before quoting any
number.

Two things do have to be right:

  - **Fingerprints.** `load_tiered_moe_placement_profile` compares
    `config_sha256` and `index_sha256` against the manifest and fails closed,
    so the profile cannot simply be copied.
  - **Slot count.** An NVFP4 expert costs 22.500 MiB resident against W4G64's
    19.125, because Marlin holds the fp8 block scales as bfloat16. The same
    HBM buys about 15% fewer hot experts, so the ported profile has to be
    trimmed or the planner will fail its audit.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trim_profile import trim, validate  # noqa: E402

W4G64_RUNTIME_BYTES = 20_054_024
NVFP4_RUNTIME_BYTES = 23_592_980


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-profile", type=Path, required=True)
    ap.add_argument("--model", required=True, help="NVFP4 checkpoint directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--headroom-gib",
        type=float,
        default=0.0,
        help="extra HBM per rank to leave free beyond the equal-bytes slot count",
    )
    args = ap.parse_args()

    from vllm.model_executor.model_loader.tiered_moe_manifest import (
        build_glm_w4a16_manifest,
    )

    manifest = build_glm_w4a16_manifest(args.model)
    if manifest.runtime_expert_bytes != NVFP4_RUNTIME_BYTES:
        print(
            f"note: manifest runtime_expert_bytes is {manifest.runtime_expert_bytes}, "
            f"expected {NVFP4_RUNTIME_BYTES}; using the manifest value"
        )
    runtime_bytes = manifest.runtime_expert_bytes

    profile = json.loads(args.source_profile.read_text())
    validate(profile)

    ep_size = profile["ep_size"]
    owners = profile["owners"]
    per_rank_before = {r: 0 for r in range(ep_size)}
    for layer_owners, layer_hot in zip(owners, profile["hot_experts"]):
        for expert_id in layer_hot:
            per_rank_before[layer_owners[expert_id]] += 1
    slots_before = per_rank_before[0]

    budget_bytes = slots_before * W4G64_RUNTIME_BYTES
    budget_bytes -= int(args.headroom_gib * (1 << 30))
    slots_after = max(budget_bytes // runtime_bytes, 1)
    drop = slots_before - slots_after
    if drop <= 0:
        raise SystemExit("NVFP4 experts are not larger; nothing to trim")

    ported, report = trim(profile, drop)
    ported["config_sha256"] = manifest.config_sha256
    ported["index_sha256"] = manifest.index_sha256
    ported["optimizer"] = (
        f"{profile['optimizer']}+ported-to-nvfp4-PLACEHOLDER-RANKING-FROM-GLM-5.2"
    )
    validate(ported)
    args.output.write_text(json.dumps(ported))

    print(
        json.dumps(
            {
                "source_profile": str(args.source_profile),
                "hot_slots_per_rank_before": slots_before,
                "hot_slots_per_rank_after": slots_after,
                "dropped_per_rank": drop,
                "w4g64_runtime_bytes": W4G64_RUNTIME_BYTES,
                "nvfp4_runtime_bytes": runtime_bytes,
                "expert_bytes_delta_pct": round(
                    100 * (runtime_bytes / W4G64_RUNTIME_BYTES - 1), 2
                ),
                "hot_bytes_per_rank_gib": round(
                    slots_after * runtime_bytes / (1 << 30), 3
                ),
                "config_sha256": manifest.config_sha256,
                "ranking": "PLACEHOLDER: GLM-5.2 ranking, router has since moved",
                "layers_touched": report["layers_touched"],
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
