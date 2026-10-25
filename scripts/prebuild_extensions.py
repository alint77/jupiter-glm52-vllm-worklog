# Build (or confirm up to date) the JIT extensions prod loads, all at once and
# off the GPUs, so a boot finds them built instead of compiling them one after
# another inside the boot (2026-10-25-boot-time: 3.8 min when their headers
# changed). Uses the caller's environment (VLLM_CACHE_ROOT, *_DEFINES), so the
# builds are the ones the server's own load() calls would make; serve.sh starts
# it in the background at boot. TORCH_CUDA_ARCH_LIST=9.0+PTX gives one_shot
# (no -gencode of its own) the flags the GPU-visible build derives.
#     prebuild_extensions.py
import os
import subprocess
import sys
import time

BUILDS = {
    "one_shot": "vllm.distributed.device_communicators.one_shot:_extension",
    "mla_glue": "vllm.model_executor.layers.mla_glue:_extension",
    "tiered_prefill": "vllm.model_executor.layers.fused_moe.tiered_prefill:_extension",
    "decode_gemm": "vllm.model_executor.layers.decode_gemm:_extension",
    "anchor_hot": "vllm.v1.attention.backends.mla.anchor_hot_cache:_extension",
    "sliced_decode": "vllm.model_executor.layers.fused_moe.tiered_decode.sliced:_library",
}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCH_CUDA_ARCH_LIST="9.0+PTX")
t0 = time.time()
procs = {}
for tag, target in BUILDS.items():
    module, fn = target.split(":")
    code = f"import {module} as m; m.{fn}()"
    procs[tag] = subprocess.Popen([sys.executable, "-c", code], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True)
failed = 0
for tag, p in procs.items():
    _, err = p.communicate()
    ok = p.returncode == 0
    failed += not ok
    print(f"prebuild {tag}: {'ok' if ok else 'FAILED'} at {time.time() - t0:.0f}s",
          flush=True)
    if not ok:
        print(err[-2000:], file=sys.stderr)
sys.exit(1 if failed else 0)
