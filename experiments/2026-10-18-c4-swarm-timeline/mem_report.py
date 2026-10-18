"""DRAM / VRAM report of one arm from mem_sampler.py (mem/mem.jsonl,
mem/numa.jsonl) and the server log's phase lines.

MemTotal / MemAvailable span the HBM NUMA nodes too (cudaMalloc'd HBM counts
as used memory of nodes 4/12/20/28), so "Grace used" is the CPU nodes' sum.

  1. startup: per phase (first log line of each), GPU used per GPU, CPU NUMA
     node used / file / anon / shmem / mlocked, vLLM process RSS
  2. serving (from the replay's t0, timings.jsonl, else server ready):
     min / median / max per quantity, per-process Pss split, VmLck / VmPin,
     CPU cores per process, GPU util / power / PCIe-C2C rx/tx
  3. 30 s bins over serving (GPU used, host used, page cache, worker RSS)
  4. numastat -p per process (last sample): MiB per NUMA node

    mem_report.py <arm out dir> [--bins 30]
"""
import argparse
import datetime
import json
import re
import statistics as st
from pathlib import Path

PHASES = [("server start", r"non-default args"),
          ("dist init", r"world_size=4|Using V2 Model Runner|init_distributed|NCCL version"),
          ("residency plan", r"Tiered MoE residency"),
          ("weights loaded", r"Model loading took"),
          ("KV allocated", r"Tiered KV main"),
          ("graphs captured", r"Graph capturing finished"),
          ("HBM check", r"observed HBM reserve"),
          ("ready", r"Application startup complete|Starting vLLM API server")]
TS = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")


def log_times(path, year):
    out = {}
    if not path.exists():
        return out
    for line in path.open(errors="ignore"):
        for name, pat in PHASES:
            if name in out or not re.search(pat, line):
                continue
            m = TS.search(line)
            if m:
                mo, d, h, mi, s = map(int, m.groups())
                out[name] = datetime.datetime(year, mo, d, h, mi, s).timestamp()
    return out


def q(v):
    v = sorted(v)
    return (v[0], st.median(v), v[-1]) if v else (0, 0, 0)


def fmt(t):
    return " / ".join(f"{x:,.0f}" for x in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--bins", type=float, default=30)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in open(d / "mem/mem.jsonl")]
    t_start = rows[0]["t"]
    year = datetime.datetime.fromtimestamp(t_start).year
    ph = log_times(d / "server.out", year)
    t0 = None
    if (d / "timings.jsonl").exists():
        t0 = json.loads(open(d / "timings.jsonl").readline()).get("t0_unix")
    t0 = t0 or ph.get("ready", rows[-1]["t"])
    cpu_nodes = sorted(int(k) for k in rows[0]["node"] if int(k) < 4)

    def grace(r):
        return sum(r["node"][str(n)]["MemUsed"] for n in cpu_nodes)

    def at(t):
        return min(rows, key=lambda r: abs(r["t"] - t))

    print(f"== {d.name}: {len(rows)} samples, {rows[-1]['t'] - t_start:.0f} s; "
          f"serving from +{t0 - t_start:.0f} s")
    print("\n1. startup (MiB)")
    print(f"{'phase':<16}{'+s':>6} | GPU used 0..3{'':>14} | Grace used cache  mlock | "
          f"CPU node used 0..3{'':>10} | RSS api / EC / W0")
    for name, _ in PHASES:
        if name not in ph:
            continue
        r = at(ph[name] + 1)
        g = " ".join(f"{x['used']:>6}" for x in r["gpu"])
        m = r["mem"]
        nu = " ".join(f"{r['node'][str(n)]['MemUsed']:>7}" for n in cpu_nodes)
        p = r["proc"]
        rss = " / ".join(str(p.get(k, {}).get("Rss", "-"))
                         for k in ("api", "EngineCore", "Worker_TP0_DCP0"))
        print(f"{name:<16}{ph[name] - t_start:>6.0f} | {g} | "
              f"{grace(r):>9} {m['Cached']:>6} {m['Mlocked']:>6} | "
              f"{nu} | {rss}")

    sv = [r for r in rows if r["t"] >= t0]
    if not sv:
        return
    print(f"\n2. serving ({len(sv)} samples, {sv[-1]['t'] - sv[0]['t']:.0f} s): min / median / max")
    for i in range(len(sv[0]["gpu"])):
        g = [r["gpu"][i] for r in sv]
        print(f"  GPU{i}: used {fmt(q([x['used'] for x in g]))} of {g[0]['total']:,} MiB | "
              f"SM util {fmt(q([x['sm'] for x in g]))} % | mem util {fmt(q([x['mem'] for x in g]))} % | "
              f"{fmt(q([x['w'] for x in g]))} W | clk {fmt(q([x['clk'] for x in g]))} | "
              f"rx {fmt(q([x.get('rx', 0) / 1e3 for x in g]))} tx {fmt(q([x.get('tx', 0) / 1e3 for x in g]))} MB/s")
        apps = {}
        for r in g:
            for pid, mb in (r.get("apps") or {}).items():
                apps.setdefault(pid, []).append(mb)
        print("        per process:", ", ".join(f"pid {p} {fmt(q(v))}" for p, v in apps.items()))
    for k in ("MemTotal", "MemAvailable", "Cached", "Shmem", "AnonPages", "Mlocked", "Mapped",
              "Committed_AS", "PageTables", "SUnreclaim"):
        print(f"  host {k:<13} {fmt(q([r['mem'][k] for r in sv]))} MiB")
    print("  Grace used (CPU nodes)", fmt(q([grace(r) for r in sv])))
    print("  HBM nodes used        ", fmt(q([sum(v["MemUsed"] for n, v in r["node"].items()
                                               if int(n) >= 4) for r in sv])))
    for n in sorted(sv[0]["node"], key=int):
        keys = ("MemUsed", "FilePages", "AnonPages", "Shmem", "Mlocked")
        print(f"  NUMA node {n:>2}: " + " | ".join(
            f"{k} {fmt(q([r['node'][n].get(k, 0) for r in sv if n in r['node']]))}" for k in keys))
    names = sorted({k for r in sv for k in r["proc"]})
    print("  per process (MiB): Rss | Pss_Anon | Pss_File | Pss_Shmem | VmLck | VmPin | threads | cores")
    for k in names:
        pr = [r["proc"][k] for r in sv if k in r["proc"]]
        if not pr:
            continue
        cs = [(r["t"], r["proc"][k].get("cpu_s")) for r in sv if k in r["proc"]]
        cores = (cs[-1][1] - cs[0][1]) / max(1e-9, cs[-1][0] - cs[0][0]) if cs[0][1] is not None else 0
        print(f"    {k:<14} " + " | ".join(fmt(q([x.get(f, 0) for x in pr])) for f in
              ("Rss", "Pss_Anon", "Pss_File", "Pss_Shmem", "VmLck", "VmPin"))
              + f" | {pr[-1].get('Threads')} | {cores:.2f}")

    print(f"\n3. serving, {a.bins:.0f} s bins (MiB): GPU0..3 used | Grace used | cache | W0..3 Rss | GPU0 SM% / W")
    b0 = sv[0]["t"]
    bins = {}
    for r in sv:
        bins.setdefault(int((r["t"] - b0) // a.bins), []).append(r)
    for b, rs in sorted(bins.items()):
        r = rs[-1]
        g = " ".join(f"{max(x['gpu'][i]['used'] for x in rs):>6}" for i in range(len(r["gpu"])))
        w = " ".join(f"{next((v.get('Rss', 0) for k, v in r['proc'].items() if k.startswith(f'Worker_TP{i}')), 0):>6}"
                     for i in range(4))
        sm = st.mean(x["gpu"][0]["sm"] for x in rs)
        pw = st.mean(x["gpu"][0]["w"] for x in rs)
        print(f"  {b * a.bins:>5.0f} | {g} | {grace(r):>7} | "
              f"{r['mem']['Cached']:>7} | {w} | {sm:>3.0f} / {pw:>3.0f}")

    nf = d / "mem/numa.jsonl"
    if nf.exists():
        last = {}
        for l in nf.open():
            r = json.loads(l)
            last[r["proc"]] = r
        print("\n4. numastat -p (last sample), MiB per node; took s")
        for k, r in sorted(last.items()):
            lines = r["numastat"].splitlines()
            hdr = next((l.split()[1::2] for l in lines if l.strip().startswith("Node")), [])
            tot = next((l.split()[1:] for l in lines if l.startswith("Total")), [])
            nz = " ".join(f"n{h}={float(v):,.0f}" for h, v in zip(hdr, tot) if float(v) >= 1)
            print(f"  {k:<16} +{r['t'] - t_start:>5.0f}s took {r['took']:.1f}s: {nz} | total {tot[-1] if tot else '-'}")


if __name__ == "__main__":
    main()
