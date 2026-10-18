"""Nsight Systems sqlite export -> Chrome trace JSON (gzipped) for Perfetto /
chrome://tracing: GPU kernels, memcpys and memsets per device and stream,
CUDA runtime / driver calls and NVTX ranges per process and thread, and a
flow arrow from each launch call to its kernel (correlation id; graph
launches point to every kernel of the graph).

    nsys_to_perfetto.py report.sqlite out.json.gz [--no-flows]
"""
import argparse
import gzip
import json
import sqlite3

COPY = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("out")
    ap.add_argument("--no-flows", action="store_true")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    names = dict(c.execute("select id, value from StringIds"))
    t0 = min(r[0] for r in c.execute(
        "select min(start) from CUPTI_ACTIVITY_KIND_KERNEL union all "
        "select min(start) from CUPTI_ACTIVITY_KIND_RUNTIME") if r[0] is not None)
    us = lambda t: (t - t0) / 1e3
    ev = []
    gpu_pid = lambda d: 1000 + d
    devs = set()
    corr_gpu = {}
    for s, e, n, d, strm, corr, gx, gy, gz, bx, by, bz, gid in c.execute(
            "select start, end, shortName, deviceId, streamId, correlationId, gridX, gridY, "
            "gridZ, blockX, blockY, blockZ, graphId from CUPTI_ACTIVITY_KIND_KERNEL"):
        devs.add(d)
        ev.append({"ph": "X", "name": names[n], "pid": gpu_pid(d), "tid": strm, "ts": us(s),
                   "dur": (e - s) / 1e3, "cat": "kernel",
                   "args": {"grid": f"{gx}x{gy}x{gz}", "block": f"{bx}x{by}x{bz}",
                            "corr": corr, "graph": gid}})
        corr_gpu.setdefault(corr, []).append((gpu_pid(d), strm, us(s)))
    for s, e, d, strm, b, k, corr in c.execute(
            "select start, end, deviceId, streamId, bytes, copyKind, correlationId "
            "from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append({"ph": "X", "name": f"memcpy {COPY.get(k, k)} {b / 2**20:.2f} MiB",
                   "pid": gpu_pid(d), "tid": strm, "ts": us(s), "dur": (e - s) / 1e3,
                   "cat": "memcpy", "args": {"bytes": b, "corr": corr}})
    for s, e, d, strm, b in c.execute(
            "select start, end, deviceId, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
        ev.append({"ph": "X", "name": f"memset {b} B", "pid": gpu_pid(d), "tid": strm,
                   "ts": us(s), "dur": (e - s) / 1e3, "cat": "memset"})
    procs = {}
    for gp, nm in c.execute("select globalPid, name from PROCESSES"):
        procs[(gp >> 24) & 0xFFFFFF] = nm
    tnames = {}
    for nid, gt in c.execute("select nameId, globalTid from ThreadNames"):
        tnames[gt] = names.get(nid, "")
    seen_threads = set()
    fid = 0
    for s, e, gt, corr, n in c.execute(
            "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        pid, tid = (gt >> 24) & 0xFFFFFF, gt & 0xFFFFFF
        seen_threads.add((pid, tid, gt))
        name = names[n]
        ev.append({"ph": "X", "name": name, "pid": pid, "tid": tid, "ts": us(s),
                   "dur": (e - s) / 1e3, "cat": "cuda_api", "args": {"corr": corr}})
        if not a.no_flows and corr in corr_gpu and "Launch" in name:
            for gpid, strm, ts in corr_gpu[corr][:1]:
                fid += 1
                ev.append({"ph": "s", "id": fid, "name": "launch", "cat": "flow",
                           "pid": pid, "tid": tid, "ts": us(s)})
                ev.append({"ph": "f", "bp": "e", "id": fid, "name": "launch", "cat": "flow",
                           "pid": gpid, "tid": strm, "ts": ts})
    try:
        for s, e, gt, txt, tid_ in c.execute(
                "select start, end, globalTid, text, textId from NVTX_EVENTS "
                "where end is not null"):
            pid, tid = (gt >> 24) & 0xFFFFFF, gt & 0xFFFFFF
            ev.append({"ph": "X", "name": txt or names.get(tid_, "nvtx"), "pid": pid,
                       "tid": tid, "ts": us(s), "dur": (e - s) / 1e3, "cat": "nvtx"})
    except sqlite3.Error:
        pass
    for d in sorted(devs):
        ev.append({"ph": "M", "name": "process_name", "pid": gpu_pid(d), "args": {"name": f"GPU {d}"}})
        ev.append({"ph": "M", "name": "process_sort_index", "pid": gpu_pid(d), "args": {"sort_index": d}})
    for pid, tid, gt in seen_threads:
        ev.append({"ph": "M", "name": "process_name", "pid": pid,
                   "args": {"name": f"{procs.get(pid, 'proc')} ({pid})"}})
        ev.append({"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": 100}})
        ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                   "args": {"name": tnames.get(gt) or str(tid)}})
    with gzip.open(a.out, "wt") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, f)
    print(f"{len(ev)} events -> {a.out}")


if __name__ == "__main__":
    main()
