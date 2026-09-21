"""Load MiMo-V2.6-Pro-RL without the tiered path and report KV geometry.

The tiered planner needs to know what the KV cache actually costs before it can
budget for it, and this model's 60 sliding-window layers may or may not be
promoted to full attention for allocation -- a 7x swing at 250K. Measure it.
"""

import os

from vllm import LLM, SamplingParams

MODEL = "/e/fscratch/profound/naeimitabiei1/models/MiMo-V2.6-Pro-RL"


def main() -> None:
    max_len = int(os.environ.get("SMOKE_MAX_LEN", "65536"))
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        max_model_len=max_len,
        max_num_seqs=1,
        cpu_offload_gb=float(os.environ.get("SMOKE_OFFLOAD_GB", "60")),
        # cpu_offload_gb routes weights through UVA, and unbound C2C runs about
        # 6x slow on this machine -- enough to turn weight load into the whole
        # job. Bind before any pinned allocation happens.
        numa_bind=True,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
    )
    out = llm.generate(
        ["What is 17*23? Answer with just the number."],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    for o in out:
        print("GENERATED:", repr(o.outputs[0].text))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
