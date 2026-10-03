"""SM clock, power and throttle reasons while the 128-token dense tile runs
back to back (w13 and w2, modes 0 / 2 / 3), sampled with NVML every 20 ms.
Is the full kernel's gap to its compute-only modes a clock (power) drop?"""
import threading
import time

import numpy as np
import pynvml
import torch

from vllm.model_executor.layers.fused_moe import tiered_prefill

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
dev = torch.device("cuda", 0)
E, N = 64, 128
REASONS = {0x4: "sw_power", 0x8: "hw_slowdown", 0x20: "sw_thermal",
           0x40: "hw_thermal", 0x80: "hw_power_brake", 0x2: "app_clocks",
           0x100: "display", 0x1: "idle"}


def sample(stop, out):
    while not stop.is_set():
        out.append((pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                    pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)))
        time.sleep(0.02)


print("power limit W:", pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)
for k, f, name in ((6144, 4096, "w13"), (2048, 6144, "w2")):
    q = torch.randint(-2**31, 2**31 - 1, (E, k // 16, f * 2), dtype=torch.int32, device=dev)
    s = ((0.5 + torch.rand((E, k // 32, f), device=dev) / 2) / 64).to(torch.bfloat16)
    k_exp = tiered_prefill.scale_exponent(s)
    x = torch.randn((N, k), dtype=torch.bfloat16, device=dev)
    for mode in (0, 2, 3):
        fn = lambda: tiered_prefill.dense(x, q, s, mode, k_exp)  # noqa: E731
        fn()
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.accelerator.synchronize()
        out, stop = [], threading.Event()
        t = threading.Thread(target=sample, args=(stop, out))
        t0 = time.time()
        reps = 0
        t.start()
        while time.time() - t0 < 4:
            g.replay()
            reps += 20
            if reps % 400 == 0:
                torch.accelerator.synchronize()
        torch.accelerator.synchronize()
        el = time.time() - t0
        stop.set()
        t.join()
        a = np.array([(c, p) for c, p, _ in out[len(out) // 4:]])
        rs = 0
        for _, _, r in out[len(out) // 4:]:
            rs |= r
        why = ",".join(v for b, v in REASONS.items() if rs & b) or "-"
        print(f"{name} mode{mode}: {el / reps * 1e6:7.1f} us/call  SM clock "
              f"{a[:, 0].mean():6.0f} MHz (min {a[:, 0].min():.0f})  power "
              f"{a[:, 1].mean():5.0f} W  throttle {why}", flush=True)
