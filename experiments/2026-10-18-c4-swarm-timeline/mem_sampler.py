"""Node memory sampler for a serving arm, from before the server starts until
killed. Every PERIOD s (mem.jsonl):
  gpu    per GPU: NVML used / total, per-process used, SM / mem util, SM clock,
         power, PCIe/C2C tx/rx KB/s
  node   /proc/meminfo (MemUsed, Cached, Shmem, AnonPages, Mlocked, ...) and
         per NUMA node MemUsed / FilePages / AnonPages / Shmem / Mlocked
         (CPU nodes 0-3, HBM nodes 4/12/20/28)
  proc   per vLLM process (api, EngineCore, Worker_TP<i>): smaps_rollup Rss,
         Pss_Anon/File/Shmem, status VmLck/VmPin, threads, CPU s
Every NUMA_PERIOD s: numastat -p per process (per-node split, numa.jsonl).

    mem_sampler.py --out DIR [--period 1] [--numa-period 30]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pynvml

NODE_KEYS = ("MemUsed", "FilePages", "AnonPages", "Shmem", "Mlocked", "Unevictable",
             "PageTables", "SUnreclaim")
MEM_KEYS = ("MemTotal", "MemAvailable", "MemFree", "Cached", "Buffers", "Shmem",
            "AnonPages", "Mlocked", "Unevictable", "Committed_AS", "PageTables",
            "Mapped", "SUnreclaim")
ROLL = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Swap")
STATUS = ("VmRSS", "VmLck", "VmPin", "VmSwap", "Threads")
CLK = os.sysconf("SC_CLK_TCK")


def kb(text, keys):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(?:Node \d+ )?(\w+):\s+(\d+)", line)
        if m and m.group(1) in keys:
            out[m.group(1)] = int(m.group(2)) // 1024  # MiB
    return out


def procs():
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            comm = (p / "comm").read_text().strip()
        except OSError:
            continue
        if "bin/vllm serve" in cmd and "nsys" not in cmd.split()[0]:
            name = "api"
        elif cmd.startswith("VLLM::") or comm.startswith("VLLM::"):
            name = (cmd.split()[0] if cmd.startswith("VLLM::") else comm)[6:]
        else:
            continue
        name = name if name not in out else f"{name}-{p.name}"
        out[name] = int(p.name)
    return out


def proc_row(pid):
    d = Path(f"/proc/{pid}")
    row = {}
    try:
        row.update(kb((d / "smaps_rollup").read_text(), ROLL))
        st = (d / "status").read_text()
        row.update(kb(st, STATUS))
        row["Threads"] = int(re.search(r"Threads:\s+(\d+)", st).group(1))
        f = (d / "stat").read_text().rsplit(")", 1)[1].split()
        row["cpu_s"] = (int(f[11]) + int(f[12])) / CLK
    except (OSError, AttributeError):
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--numa-period", type=float, default=30.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    nodes = sorted(int(n.name[4:]) for n in Path("/sys/devices/system/node").glob("node[0-9]*"))
    f = open(out / "mem.jsonl", "a")
    fn = open(out / "numa.jsonl", "a")
    last_numa = 0.0
    while True:
        t = time.time()
        row = {"t": t, "gpu": [], "node": {}, "proc": {}}
        for i, h in enumerate(hs):
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            g = {"i": i, "used": m.used >> 20, "total": m.total >> 20, "sm": u.gpu,
                 "mem": u.memory,
                 "clk": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                 "w": pynvml.nvmlDeviceGetPowerUsage(h) // 1000}
            try:
                g["tx"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                g["rx"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            except pynvml.NVMLError:
                pass
            try:
                g["apps"] = {p.pid: (p.usedGpuMemory or 0) >> 20
                             for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
            except pynvml.NVMLError:
                pass
            row["gpu"].append(g)
        row["mem"] = kb(open("/proc/meminfo").read(), MEM_KEYS)
        for n in nodes:
            v = kb(open(f"/sys/devices/system/node/node{n}/meminfo").read(), NODE_KEYS)
            if v.get("MemUsed") or n < 4:
                row["node"][n] = v
        ps = procs()
        row["proc"] = {k: dict(proc_row(p), pid=p) for k, p in ps.items()}
        f.write(json.dumps(row) + "\n")
        f.flush()
        if ps and t - last_numa >= a.numa_period:
            last_numa = t
            for k, p in ps.items():
                try:
                    s = subprocess.run(["numastat", "-p", str(p)], capture_output=True,
                                       text=True, timeout=60).stdout
                    fn.write(json.dumps({"t": t, "proc": k, "pid": p,
                                         "took": time.time() - t, "numastat": s}) + "\n")
                except subprocess.TimeoutExpired:
                    pass
            fn.flush()
        time.sleep(max(0.0, a.period - (time.time() - t)))


if __name__ == "__main__":
    main()
