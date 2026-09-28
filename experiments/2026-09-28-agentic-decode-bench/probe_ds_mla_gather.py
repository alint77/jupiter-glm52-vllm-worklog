#!/usr/bin/env python3
"""Can the MHA prefill-context gathers read an fp8_ds_mla cache?

Writes known bf16 (kv_c, k_pe) into a [blocks, 64, 656] fp8_ds_mla cache with
the production writer (concat_and_cache_mla), then reads it back through
  (a) gather_and_maybe_dequant_cache(kv_cache_dtype="fp8_ds_mla")  -- the
      non-DCP MHA context path (_compute_prefill_context)
  (b) cp_gather_and_upconvert_fp8_kv_cache  -- the sparse backend's own reader
and reports the error of each against the originals (fp8 rounding only is
~3% relative on kv_c; k_pe is stored bf16 and should be exact).
"""

import torch

from vllm import _custom_ops as ops

torch.manual_seed(0)
dev = torch.device("cuda:0")
BLOCK, T = 64, 300
kv_c = (torch.randn(T, 512, device=dev) * 0.5).to(torch.bfloat16)
k_pe = torch.randn(T, 64, device=dev).to(torch.bfloat16)
ref = torch.cat([kv_c, k_pe], -1).float()
nblocks = (T + BLOCK - 1) // BLOCK + 2
cache = torch.zeros(nblocks, BLOCK, 656, dtype=torch.uint8, device=dev)
perm = torch.randperm(nblocks, device=dev)[: (T + BLOCK - 1) // BLOCK].int()
slots = (perm.repeat_interleave(BLOCK)[:T].long() * BLOCK
         + torch.arange(T, device=dev) % BLOCK)
scale = torch.ones(1, device=dev)
ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, "fp8_ds_mla", scale)
block_table = perm.view(1, -1)


def report(name, out):
    got = out.float()
    e_c = ((got[:, :512] - ref[:, :512]).norm() / ref[:, :512].norm()).item()
    e_p = ((got[:, 512:] - ref[:, 512:]).norm() / ref[:, 512:].norm()).item()
    print(f"{name}: rel err kv_c {e_c:.4f}, k_pe {e_p:.4f}, "
          f"finite {bool(torch.isfinite(got).all())}")


out_b = torch.zeros(T, 576, dtype=torch.bfloat16, device=dev)
ops.cp_gather_and_upconvert_fp8_kv_cache(
    cache, out_b, block_table, torch.tensor([T], dtype=torch.int32, device=dev),
    torch.tensor([0], dtype=torch.int32, device=dev), 1)
report("(b) cp_gather_and_upconvert_fp8_kv_cache", out_b)

out_a = torch.zeros(T, 576, dtype=torch.bfloat16, device=dev)
try:
    ops.gather_and_maybe_dequant_cache(
        src_cache=cache, dst=out_a, block_table=block_table,
        cu_seq_lens=torch.tensor([0, T], dtype=torch.int32, device=dev),
        token_to_seq=torch.zeros(T, dtype=torch.int32, device=dev),
        num_tokens=T, kv_cache_dtype="fp8_ds_mla", scale=scale,
        seq_starts=torch.tensor([0], dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    report("(a) gather_and_maybe_dequant_cache('fp8_ds_mla')", out_a)
except Exception as error:  # the question is whether it runs at all
    print(f"(a) gather_and_maybe_dequant_cache('fp8_ds_mla') raised: {error}")

# The fix: MLACommonImpl._gather_ds_mla_context, two requests, a context chunk
# that starts mid-sequence (seq_starts) as chunked prefill produces.
import types  # noqa: E402

from vllm.model_executor.layers.attention.mla_attention import (  # noqa: E402
    MLACommonImpl,
)

fake = types.SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
T2 = 200
kv_c2 = (torch.randn(T2, 512, device=dev) * 0.5).to(torch.bfloat16)
k_pe2 = torch.randn(T2, 64, device=dev).to(torch.bfloat16)
nb2 = (T2 + BLOCK - 1) // BLOCK
perm2 = (torch.randperm(nb2, device=dev) + nblocks).int()
cache2 = torch.zeros(nblocks + nb2, BLOCK, 656, dtype=torch.uint8, device=dev)
cache2[:nblocks] = cache
slots2 = perm2.repeat_interleave(BLOCK)[:T2].long() * BLOCK + torch.arange(T2, device=dev) % BLOCK
ops.concat_and_cache_mla(kv_c2, k_pe2, cache2, slots2, "fp8_ds_mla", scale)
width = max(block_table.shape[1], nb2)
bt = torch.zeros(2, width, dtype=torch.int32, device=dev)
bt[0, : block_table.shape[1]] = block_table[0]
bt[1, :nb2] = perm2
starts = torch.tensor([100, 64], dtype=torch.int32, device=dev)
lens = [150, 120]  # tokens [100, 250) of request 0, [64, 184) of request 1
cu = torch.tensor([0, lens[0], lens[0] + lens[1]], dtype=torch.int32, device=dev)
ws = torch.zeros(sum(lens), 576, dtype=torch.bfloat16, device=dev)
MLACommonImpl._gather_ds_mla_context(fake, cache2, ws, bt, cu, 2, starts, sum(lens))
ref2 = torch.cat([torch.cat([kv_c, k_pe], -1)[100:250],
                  torch.cat([kv_c2, k_pe2], -1)[64:184]]).float()
got = ws.float()
e_c = ((got[:, :512] - ref2[:, :512]).norm() / ref2[:, :512].norm()).item()
e_p = ((got[:, 512:] - ref2[:, 512:]).norm() / ref2[:, 512:].norm()).item()
print(f"(fix) _gather_ds_mla_context, 2 requests, mid-sequence starts: "
      f"rel err kv_c {e_c:.4f}, k_pe {e_p:.4f}, finite {bool(torch.isfinite(got).all())}")
