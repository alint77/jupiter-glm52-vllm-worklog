# One shard through the tp_sliced loader's per-expert slicing (rank `rank` of 4:
# gate/up rows, down columns, then .t().contiguous()), read either through
# safetensors' mmap (as the loader does) or as one read() of the whole file.
#     shard_probe.py <mmap|eager> <rank> <shard path>
import sys, time
import torch
from safetensors import safe_open
from safetensors.torch import load

mode, rank, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
inter = 512
t0 = time.perf_counter()
if mode == "eager":
    with open(path, "rb") as f:
        tensors = load(f.read())
    items = tensors.items()
else:
    f = safe_open(path, framework="pt")
    items = ((k, f.get_tensor(k)) for k in f.keys())
t_open = time.perf_counter() - t0
n = moved = 0
for name, t in items:
    if ".experts." not in name or t.dim() != 2:
        continue
    if "gate_proj" in name or "up_proj" in name:
        t = t[rank * inter:(rank + 1) * inter]
    elif "down_proj" in name:
        div = 8 if "packed" in name else 32
        t = t[:, rank * inter // div:(rank + 1) * inter // div]
    out = t.t().contiguous()
    n += 1
    moved += out.numel() * out.element_size()
dt = time.perf_counter() - t0
print(f"{mode} rank{rank} {path.rsplit('/', 1)[1]} tensors={n} sliced={moved / 2**20:.0f}MiB "
      f"read_open={t_open:.2f}s total={dt:.2f}s")
