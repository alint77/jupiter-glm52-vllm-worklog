"""Compare exa_project1 (where the checkpoint lives) with exa_fscratch.

Checkpoint load is the slowest part of every job here: 86 shards of 5.37 GB at
~10.5-11.8 s each, about 490 MB/s. The question is whether that is the
filesystem or the access pattern, so this measures three regimes:

* O_DIRECT queue-depth-1 - latency bound, no readahead, no page cache. This is
  the closest proxy for mmap page-fault driven reads, which is how safetensors
  actually loads.
* buffered sequential - readahead friendly, the best case a streaming copy sees.
* parallel O_DIRECT - whether concurrency recovers what QD1 loses.
* a real `safetensors` load of one shard, which is the thing we care about.

Every measurement uses a shard that has not been read yet in this process, and
the O_DIRECT paths bypass the page cache entirely, so a 858 GB node cannot
flatter the second filesystem measured.
"""

import argparse
import json
import os
import statistics
import subprocess
import time

SHARDS = [f"model-{i:05d}-of-00081.safetensors" for i in range(1, 9)]


def dd_read(path, count=512, bs="4M", direct=True):
    cmd = ["dd", f"if={path}", "of=/dev/null", f"bs={bs}", f"count={count}"]
    if direct:
        cmd.append("iflag=direct")
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, check=True)
    dt = time.perf_counter() - t0
    nbytes = count * 4 * 1024 * 1024 if bs == "4M" else 0
    return nbytes / dt / 1e9, dt


def dd_parallel(paths, count=256, bs="4M"):
    procs = []
    t0 = time.perf_counter()
    for p in paths:
        procs.append(subprocess.Popen(
            ["dd", f"if={p}", "of=/dev/null", f"bs={bs}", f"count={count}",
             "iflag=direct"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL))
    for pr in procs:
        pr.wait()
    dt = time.perf_counter() - t0
    total = len(paths) * count * 4 * 1024 * 1024
    return total / dt / 1e9, dt


def mmap_fault_read(path, nbytes=2 << 30, stride=4096):
    """Touch one byte per page, the way an mmap'd tensor load faults them in."""
    import mmap
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        total = min(nbytes, len(mm))
        acc = 0
        for off in range(0, total, stride):
            acc += mm[off]
        mm.close()
    dt = time.perf_counter() - t0
    return total / dt / 1e9, dt


def safetensors_load(path):
    from safetensors import safe_open
    t0 = time.perf_counter()
    n = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            n += f.get_tensor(k).numel()
    dt = time.perf_counter() - t0
    return os.path.getsize(path) / dt / 1e9, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="/e/project1/profound/alint77/"
                    "models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887")
    ap.add_argument("--fscratch-dir",
                    default="/e/fscratch/profound/naeimitabiei1/fsbench")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = {"project1": args.project_dir, "fscratch": args.fscratch_dir}
    res = {}
    for name, d in targets.items():
        paths = [os.path.join(d, s) for s in SHARDS]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"{name}: missing {len(missing)} shards, skipping")
            continue
        r = {}
        gb, dt = dd_read(paths[0])
        r["odirect_qd1_GBps"] = gb
        print(f"{name:>10}  O_DIRECT QD1        {gb:6.2f} GB/s  ({dt:.1f} s)")

        gb, dt = dd_read(paths[1], direct=False)
        r["buffered_GBps"] = gb
        print(f"{name:>10}  buffered sequential {gb:6.2f} GB/s  ({dt:.1f} s)")

        for n in (4, 8):
            gb, dt = dd_parallel(paths[:n])
            r[f"odirect_x{n}_GBps"] = gb
            print(f"{name:>10}  O_DIRECT x{n:<2}         {gb:6.2f} GB/s  ({dt:.1f} s)")

        gb, dt = mmap_fault_read(paths[2])
        r["mmap_fault_GBps"] = gb
        print(f"{name:>10}  mmap page-fault     {gb:6.2f} GB/s  ({dt:.1f} s)")

        try:
            gb, dt = safetensors_load(paths[3])
            r["safetensors_GBps"] = gb
            r["safetensors_shard_s"] = dt
            print(f"{name:>10}  safetensors shard   {gb:6.2f} GB/s  ({dt:.1f} s)")
        except Exception as e:
            print(f"{name:>10}  safetensors failed: {e}")
        res[name] = r
        print()

    if "project1" in res and "fscratch" in res:
        print("speedup fscratch / project1:")
        for k in res["project1"]:
            if k.endswith("GBps"):
                a, b = res["project1"][k], res["fscratch"][k]
                print(f"  {k:<22} {b / a:5.2f}x   ({a:.2f} -> {b:.2f} GB/s)")
        if "safetensors_shard_s" in res["project1"]:
            a = res["project1"]["safetensors_shard_s"]
            b = res["fscratch"]["safetensors_shard_s"]
            print(f"\n  one shard: {a:.1f} s -> {b:.1f} s")
            print(f"  86 shards, single stream: {86 * a / 60:.1f} min -> "
                  f"{86 * b / 60:.1f} min")

    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
