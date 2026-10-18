# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic: the routed expert ids of every sliced decode MoE call, recorded
inside the CUDA graph into a per-step ring (VLLM_ROUTE_RECORD=<steps>). The
step counter advances at the first routed layer, so draft passes (MTP layer
only) land in the current verify step's row. The worker dumps the ring at each
profiler start and stop to VLLM_ROUTE_RECORD_DIR."""

import os

import torch

from vllm.model_executor.models.utils import extract_layer_index

STEPS = int(os.environ.get("VLLM_ROUTE_RECORD", "0"))
FIRST = int(os.environ.get("VLLM_ROUTE_RECORD_FIRST", "3"))
LAYERS = 96
WIDTH = 512
_state: dict = {}


def record(layer, topk_ids, hot_map, cold_map) -> None:
    if not STEPS or topk_ids.numel() > WIDTH:
        return
    li = extract_layer_index(layer.layer_name)
    s = _state.get("s")
    if s is None:
        dev = topk_ids.device
        s = _state["s"] = dict(
            ids=torch.full((STEPS, LAYERS, WIDTH), 255, dtype=torch.uint8, device=dev),
            m=torch.zeros(STEPS, dtype=torch.int16, device=dev),
            ctr=torch.zeros(1, dtype=torch.int64, device=dev),
            maps={},
        )
    s["maps"].setdefault(li, (hot_map, cold_map))
    if li == FIRST:
        s["ctr"].add_(1)
    slot = s["ctr"] % STEPS
    flat = topk_ids.reshape(-1)
    row = torch.full((1, WIDTH), 255, dtype=torch.uint8, device=flat.device)
    row[0, : flat.numel()] = torch.where(flat < 0, 255, flat).to(torch.uint8)
    s["ids"][:, li].index_copy_(0, slot, row)
    if li == FIRST:
        s["m"].index_fill_(0, slot, topk_ids.shape[0])


def dump(tag: str, rank: int) -> None:
    s = _state.get("s")
    out = os.environ.get("VLLM_ROUTE_RECORD_DIR")
    if s is None or not out:
        return
    torch.cuda.synchronize()
    os.makedirs(out, exist_ok=True)
    torch.save(
        dict(
            ids=s["ids"].cpu(),
            m=s["m"].cpu(),
            ctr=int(s["ctr"].item()),
            maps={li: ((h >= 0).cpu(), (c >= 0).cpu()) for li, (h, c) in s["maps"].items()},
        ),
        f"{out}/route.r{rank}.{tag}.pt",
    )
