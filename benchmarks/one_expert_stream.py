#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from compressed_tensors.quantization import QuantizationArgs
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.model_loader.tiered_moe_conversion import (
    OneExpertCheckpointStager,
    convert_glm_w4a16_expert,
)
from vllm.model_executor.model_loader.tiered_moe_planner import (
    LayerExpertPlacement,
)
from vllm.model_executor.model_loader.tiered_moe_storage import (
    GLM_MARLIN_COMPONENTS,
    allocate_layer_expert_storage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--numa-node", type=int, default=0)
    return parser.parse_args()


def load_expert(args: argparse.Namespace):
    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())
    names = [
        f"{prefix}.{projection}.{component}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for component in ("weight_packed", "weight_scale", "weight_shape")
    ]
    shards = {index["weight_map"][name] for name in names}
    if len(shards) != 1:
        raise ValueError("One expert must be contained in one safetensors shard")
    shard = str(Path(args.model) / shards.pop())
    stager = OneExpertCheckpointStager()
    bundle = None
    with safe_open(shard, framework="pt") as checkpoint:
        for name in names:
            bundle = stager.add(name, checkpoint.get_tensor(name))
    stager.finish()
    if bundle is None:
        raise AssertionError("Expert checkpoint bundle is incomplete")
    return bundle


def main() -> None:
    args = parse_args()
    torch.accelerator.set_device_index(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.accelerator.reset_peak_memory_stats()

    start = time.perf_counter()
    bundle = load_expert(args)
    checkpoint_loaded = time.perf_counter()
    layer = SimpleNamespace(intermediate_size_per_partition=2048)
    quantization = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        group_size=128,
        strategy="group",
        actorder="static",
    )
    quant_method = SimpleNamespace(
        wna16_backend=WNA16MoEBackend.MARLIN,
        actorder="static",
        group_size=128,
        weight_quant=quantization,
        marlin_input_dtype=torch.bfloat16,
    )
    final = convert_glm_w4a16_expert(bundle, layer, quant_method, device)
    torch.accelerator.synchronize()
    converted_at = time.perf_counter()
    mismatches = []
    for component in GLM_MARLIN_COMPONENTS:
        tensor = final[component.name]
        if tensor.shape != component.shape or tensor.dtype != component.dtype:
            mismatches.append(
                f"{component.name}: got {tensor.shape}/{tensor.dtype}, "
                f"expected {component.shape}/{component.dtype}"
            )
    if mismatches:
        raise ValueError("; ".join(mismatches))

    storage = allocate_layer_expert_storage(
        LayerExpertPlacement(args.layer, (), (args.expert,)),
        args.device_index,
        args.numa_node,
    )
    storage.cold.copy_expert_from(args.expert, final)
    torch.accelerator.synchronize()
    committed = time.perf_counter()
    correct = all(
        torch.equal(storage.cold.components[component.name][0], final[component.name])
        for component in GLM_MARLIN_COMPONENTS
    )
    audit = storage.cold.grace_allocation.audit_numa(samples=256, strict=True)
    result = {
        "checkpoint_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in bundle.components.values()
        ),
        "final_bytes": storage.cold.num_bytes,
        "checkpoint_read_seconds": checkpoint_loaded - start,
        "marlin_conversion_seconds": converted_at - checkpoint_loaded,
        "final_commit_seconds": committed - converted_at,
        "peak_hbm_bytes": torch.accelerator.max_memory_allocated(),
        "correct": correct,
        "numa_local_fraction": audit.local_fraction,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
