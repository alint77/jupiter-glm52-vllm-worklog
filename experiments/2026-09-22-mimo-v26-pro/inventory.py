"""Reconcile MiMo V2.6's non-routed tensors against the manifest.

Runs as a job: the manifest walks 130 shard headers into 160,040 entries and
the classifier makes a second pass, which the login node kills.
"""

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_mimo_v26_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_non_routed import (
    build_mimo_v26_non_routed_runtime_inventory,
)

MODEL = "/e/fscratch/profound/naeimitabiei1/models/MiMo-V2.6-Pro-RL"
G = 1024**3


def main() -> None:
    manifest = build_mimo_v26_manifest(MODEL)
    inventory = build_mimo_v26_non_routed_runtime_inventory(manifest, tp_size=4)
    print("  placement counts     :", inventory.placement_counts)
    print("  non-routed checkpoint: %7.2f GiB" % (inventory.checkpoint_bytes / G))
    print("    replicated         : %7.2f GiB" % (inventory.replicated_bytes / G))
    print(
        "    tp-sharded ckpt    : %7.2f GiB"
        % (inventory.tp_sharded_checkpoint_bytes / G)
    )
    print(
        "    tp-sharded runtime : %7.2f GiB"
        % (inventory.tp_sharded_runtime_bytes / G)
    )
    print(
        "    dropped (mtp)      : %7.2f GiB"
        % (inventory.dropped_checkpoint_bytes / G)
    )
    print("  RUNTIME PER RANK     : %7.2f GiB" % (inventory.runtime_bytes_per_rank / G))
    experts = manifest.rank_runtime_expert_bytes(ep_size=4, ep_rank=0)
    print("  experts per rank     : %7.2f GiB" % (experts / G))
    print("  KV @250K c=1         : %7.2f GiB" % (19_208_601_600 / G))
    print("INVENTORY OK")


if __name__ == "__main__":
    main()
