"""Is the working-tree prefill MoE kernel bit-identical to a committed one?
REF_REV: git revision of the reference (default HEAD). Real GLM-5.3 routing at 2048 / 4096
tokens (bench_moe_real tiers), schedule 0, plus a short ragged chunk."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm.model_executor.layers.fused_moe import tiered_prefill as tp  # noqa: E402
import real_routing  # noqa: E402
import bench_moe_real as b  # noqa: E402

rev = os.environ.get("REF_REV", "HEAD")
src = "vllm/model_executor/layers/fused_moe/tiered_prefill/tiered_prefill.cu"
d = Path(os.environ["VLLM_CACHE_ROOT"]) / "torch_extensions" / f"tp_ref_{rev}"
d.mkdir(parents=True, exist_ok=True)
# compute nodes have no git: write the source from the login node first
if not (d / "tiered_prefill.cu").exists():
    (d / "tiered_prefill.cu").write_text(
        subprocess.check_output(["git", "show", f"{rev}:{src}"], text=True))
ref = load(name=f"tp_ref_{rev}", sources=[str(d / "tiered_prefill.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"],
           extra_ldflags=["-lcuda"], build_directory=str(d))
new = tp._extension()
for chunk in (2048, 4096, 777):
    smp = real_routing.samples(max(chunk, 2048), 2, seed=chunk)[1]
    ids = torch.from_numpy(smp.topk_ids[:chunk]).to(b.dev)
    wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
    x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev)
    local = np.flatnonzero(smp.local_map >= 0)
    hm = torch.full((256,), -1, dtype=torch.int32)
    cm = torch.full((256,), -1, dtype=torch.int32)
    hm[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32)
    cm[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
    hm, cm = hm.to(b.dev), cm.to(b.dev)
    outs = []
    for ext in (ref, new):
        tp._extension = lambda e=ext: e  # noqa: E731
        outs.append(tp.tiered_prefill_moe(x, ids, wts, hm, cm, b.hot, b.cold, b.k_exp, 0))
    print(f"{chunk} tokens: bit-identical {torch.equal(outs[0], outs[1])}, "
          f"max |diff| {(outs[0].float() - outs[1].float()).abs().max().item():.3g}", flush=True)
