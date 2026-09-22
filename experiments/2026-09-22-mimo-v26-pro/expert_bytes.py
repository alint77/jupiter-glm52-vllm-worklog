"""Measure one MiMo-V2.6 MoE layer's runtime expert bytes.

The tiered planner budgets in `runtime_expert_bytes` -- the post-repack layout,
not the checkpoint layout. GLM's is 20.3 MiB/expert under
`vllm_marlin_static_w4a16`; MiMo stores mxfp4 and the log says it selects the
MARLIN mxfp4 backend, so its runtime size is a different number that nothing in
the tree records. Every tiered budget for this model depends on it, so measure
it rather than derive it from the packing.

Also exercises the Marlin mxfp4 repack on a single layer, which is where the
four non-tiered smoke launches died without a traceback.
"""

import json
import os

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs

MODEL = "/e/fscratch/profound/naeimitabiei1/models/MiMo-V2.6-Pro-RL"


def _param_bytes(module: torch.nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, p in module.named_parameters(recurse=True):
        out[name] = p.data.numel() * p.data.element_size()
    for name, b in module.named_buffers(recurse=True):
        out.setdefault(name, b.numel() * b.element_size())
    return out


def _param_shapes(module: torch.nn.Module) -> dict[str, str]:
    """Shape and dtype per tensor -- equal byte counts either side of the
    repack do not imply equal layouts, and the tiered storage specs need the
    resident one."""
    out: dict[str, str] = {}
    for name, p in module.named_parameters(recurse=True):
        out[name] = f"{tuple(p.data.shape)} {p.data.dtype}"
    for name, b in module.named_buffers(recurse=True):
        out.setdefault(name, f"{tuple(b.shape)} {b.dtype}")
    return out


def main() -> None:
    model = os.environ.get("PROBE_MODEL", MODEL)
    cfg = json.load(open(os.path.join(model, "config.json")))

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")

    vllm_config: VllmConfig = EngineArgs(
        model=model, trust_remote_code=True, load_format="dummy",
        max_model_len=4096, enforce_eager=True,
    ).create_engine_config()

    from vllm.model_executor.layers.fused_moe import FusedMoE

    # initialize_model_parallel reads get_current_vllm_config(), so the whole
    # setup has to sit inside the context, not just the layer construction.
    torch.cuda.set_device(0)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method="env://", local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        # Construct on CUDA: gptq_marlin_repack is a CUDA-only op, so weights
        # created on CPU make process_weights_after_loading raise
        # NotImplementedError instead of producing the runtime layout.
        with torch.device("cuda"):
            experts = FusedMoE(
                num_experts=cfg["n_routed_experts"],
                top_k=cfg["num_experts_per_tok"],
                hidden_size=cfg["hidden_size"],
                intermediate_size=cfg["moe_intermediate_size"],
                renormalize=cfg.get("norm_topk_prob", True),
                quant_config=vllm_config.quant_config,
                prefix="model.layers.1.mlp.experts",
                use_grouped_topk=True,
                num_expert_group=cfg.get("n_group"),
                topk_group=cfg.get("topk_group"),
                scoring_func="sigmoid",
            )
        n = cfg["n_routed_experts"]
        before = _param_bytes(experts)
        tot_b = sum(before.values())
        print("== created (checkpoint layout) ==")
        for k, v in sorted(before.items()):
            print("   %-46s %10.2f MiB" % (k, v / 1024**2))
        print("   TOTAL %.2f MiB for %d experts -> %.3f MiB/expert"
              % (tot_b / 1024**2, n, tot_b / 1024**2 / n))
        for k, v in sorted(_param_shapes(experts).items()):
            print("   SHAPE-BEFORE %-40s %s" % (k, v))

        # The FusedMoE wrapper's .quant_method is a MoERunner; the real method
        # lives on the RoutedExperts child. Walk the tree the way the loader
        # does rather than guess the attribute path.
        done = 0
        for sub in experts.modules():
            qm = getattr(sub, "quant_method", None)
            fn = getattr(qm, "process_weights_after_loading", None)
            if callable(fn):
                print("   process_weights_after_loading on %s (%s)"
                      % (type(sub).__name__, type(qm).__name__))
                fn(sub)
                done += 1
        print("   ran on %d submodule(s)" % done)
        torch.cuda.synchronize()
        after = _param_bytes(experts)
        tot_a = sum(after.values())
        print("== after process_weights_after_loading (RUNTIME layout) ==")
        for k, v in sorted(after.items()):
            print("   %-46s %10.2f MiB" % (k, v / 1024**2))
        print("   TOTAL %.2f MiB for %d experts" % (tot_a / 1024**2, n))
        for k, v in sorted(_param_shapes(experts).items()):
            print("   SHAPE-AFTER  %-40s %s" % (k, v))
        print()
        print("RUNTIME_EXPERT_BYTES %d" % (tot_a // n))
        print("   = %.3f MiB/expert   (GLM w4a16 reference: 20.3)"
              % (tot_a / 1024**2 / n))
        print("PROBE OK")


if __name__ == "__main__":
    main()
