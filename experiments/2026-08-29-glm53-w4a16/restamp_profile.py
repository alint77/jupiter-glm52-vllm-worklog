#!/usr/bin/env python3
"""Re-fingerprint a placement profile onto another checkpoint.

The placement itself is checkpoint-independent -- owners, hot experts and
replicas are expressed in expert ids -- but the profile carries the
`config_sha256`/`index_sha256` of the checkpoint it was built against and the
loader fails closed on a mismatch. This restamps those two fields and nothing
else, so an A/B across quantizers holds the ranking genuinely fixed.

Unlike `port_profile.py` it does not trim slots and does not label the result a
placeholder: the ranking here is the real GLM-5.3 one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-08-28-nvfp4-tiered"))
from trim_profile import validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-profile", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--note", default="", help="appended to the optimizer string")
    args = ap.parse_args()

    from vllm.model_executor.model_loader.tiered_moe_manifest import (
        build_glm_w4a16_manifest,
    )

    manifest = build_glm_w4a16_manifest(args.model)
    profile = json.loads(args.source_profile.read_text())
    validate(profile)

    slots = {}
    for layer_owners, layer_hot in zip(profile["owners"], profile["hot_experts"]):
        for expert_id in layer_hot:
            rank = layer_owners[expert_id]
            slots[rank] = slots.get(rank, 0) + 1

    profile["config_sha256"] = manifest.config_sha256
    profile["index_sha256"] = manifest.index_sha256
    if args.note:
        profile["optimizer"] = f"{profile['optimizer']}+{args.note}"
    validate(profile)
    args.output.write_text(json.dumps(profile))

    print(json.dumps({
        "output": str(args.output),
        "hot_slots_per_rank": slots,
        "runtime_expert_bytes": manifest.runtime_expert_bytes,
        "group_size": manifest.group_size,
        "hot_bytes_per_rank_gib": {
            r: round(n * manifest.runtime_expert_bytes / (1 << 30), 2)
            for r, n in sorted(slots.items())
        },
        "optimizer": profile["optimizer"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
