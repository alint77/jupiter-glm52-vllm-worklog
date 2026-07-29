"""Print the Grace NUMA node paired with a GPU.

Never hardcode this and never read the GPU's own `numa_node` from sysfs: on
GH200 the GPU's NUMA node is its HBM, which has no CPUs, so sysfs returns a node
that no Grace page can ever land on. vLLM's platform helper handles that by
falling back to CPU affinity, which is what the tiered runtime itself uses.
"""

import sys

from vllm.platforms import current_platform

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
node = current_platform.get_device_numa_node(index)
if node is None:
    raise SystemExit(f"could not determine the NUMA node paired with GPU {index}")
print(node)
