#!/usr/bin/env python3
"""Run the GLM routing-profile tools on MiMo-V2.6 traces.

`traces_to_manifest.py` and `optimize_routing_profile.py` hard-code GLM-5's
routed layers (3..77) and expert count (256) as module constants. This rebinds
them from the MiMo checkpoint manifest (layers 1..69, 384 experts) and hands
over to the tool's own `main`, so the two stay one implementation.

    mimo_profile.py manifest --trace-dir DIR [tool args]
    mimo_profile.py optimize --trace-dir DIR --model PATH [tool args]
"""

import os
import sys
from pathlib import Path

REPO = Path("/e/project1/profound/alint77/vllm")
MODEL = os.environ.get(
    "MIMO_MODEL", f"/e/fscratch/profound/{os.environ['USER']}/models/MiMo-V2.6-Pro-RL"
)
sys.path[:0] = [
    str(REPO / "agent_space/benchmarks"),
    str(REPO / "agent_space/experiments/2026-08-29-glm53-routing-capture"),
]

from vllm.model_executor.model_loader.tiered_moe_manifest import (  # noqa: E402
    build_tiered_moe_manifest,
)


def main() -> None:
    tool, sys.argv[1:] = sys.argv[1], sys.argv[2:]
    manifest = build_tiered_moe_manifest(MODEL)
    if tool == "manifest":
        import traces_to_manifest as module
    elif tool == "optimize":
        import optimize_routing_profile as module

        module.NUM_EXPERTS = manifest.num_experts
        module.EXPERTS_PER_RANK = manifest.num_experts // module.EP_SIZE
        module.build_glm_w4a16_manifest = build_tiered_moe_manifest
    else:
        raise SystemExit(f"unknown tool {tool!r}")
    module.ROUTED_LAYERS = tuple(manifest.routed_layers)
    module.main()


if __name__ == "__main__":
    main()
