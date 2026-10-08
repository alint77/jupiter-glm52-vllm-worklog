#!/usr/bin/env python3
"""Op sequence, stream overlap and SOL for the dd3-w768 decode windows.

Prod serve.sh (DFlash2 k=7 captured draft, DCP4, c=1, W4A16 g32, 3,726 hot
experts, sparse-decode width 768). Steps and phase attribution reuse
../2026-09-26-mimo-decode-profile/analyze.py (launch-correlation model).

    dive4_sol.py <trace-window-dir> [--seq] [--sol]

--seq prints the kernel-by-kernel sequence of the median step's target graph
(skip layer ~39 and anchor layer 42) plus the between-graph regions.
--sol prints per-kernel speed-of-light analysis (HBM 3.64 TB/s, C2C 421 GB/s,
630 TFLOPS bf16) and the cross-rank collective wire/wait split.
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402
import gzip
import json as _json


def _load_us(path):
    """This torch's traces carry ts/dur in us; analyze.py divides by 1000 on
    purpose and operates in ms (its own millisecond reports are correct).
    dive4_sol.py does per-kernel arithmetic in us, so it replaces the loader
    with a no-op us one; analyze's own outputs are unaffected."""
    with gzip.open(path, "rt") as handle:
        events = [e for e in _json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    for event in events:
        event["t"] = event["ts"]
        event["end"] = event["t"] + event.get("dur", 0)
    return events


A.load = _load_us

US = 1e3  # us -> ms


def med_step(rows):
    return min(rows, key=lambda r: abs(st.median(r2["period"] for r2 in rows) - r["period"]))


def short(name, n=78):
    s = name.split("<")[0].split("(")[0].replace("void ", "").replace("vllm::", "").replace("at::native::", "")
    return (s[:n] + "..") if len(s) > n else s


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


# ---------------------------------------------------------------- sequences
def layer_of(step, k):
    flash = sorted((o for o in step["phases"]["target"]
                    if "flash_fwd_splitkv_mla_fp8_sparse" in o["name"]), key=lambda e: e["t"])
    return flash[k], flash[k + 1] if k + 1 < len(flash) else None


def print_seq(step, k, title):
    a, b = layer_of(step, k)
    end = b["t"] if b else a["end"] + 260.0
    main = collections.Counter(o["args"]["stream"] for o in step["phases"]["target"]).most_common(1)[0][0]
    rows = []
    for ph, ops in step["phases"].items():
        for o in ops:
            if o["end"] > a["t"] and o["t"] < end:
                rows.append((o["t"], o["end"], ph, o))
    rows.sort()
    print(f"\n--- {title}: layer ordinal {k}, {end - a['t']:.1f} us, main stream {main}")
    print(f"{'rel us':>8} {'dur':>6} {'ovl':>4} stream  kernel")
    prev_end = a["t"]
    for t, e, ph, o in rows:
        ov = "+" if t < prev_end - 0.30 else " "  # >0.3us overlap with anything before
        prev_end = max(prev_end, e)
        print(f"{t - a['t']:8.1f} {e - t:6.1f} {ov:>4} {o['args']['stream']:>6}  "
              f"{ph[:6]:6} {short(o['name'])}")


def phase_skeleton(step):
    print(f"\n--- step phases (period {step['period'] / US:.2f} ms)")
    for ph in ("target", "logits", "draft", "host-side"):
        ops = step["phases"].get(ph, [])
        if not ops:
            print(f"  {ph:10} -")
            continue
        span = (max(o["end"] for o in ops) - min(o["t"] for o in ops)) / US
        busy = A.union(ops) / US
        print(f"  {ph:10} {len(ops):5d} kernels  span {span:6.2f} ms  busy {busy:6.2f} ms"
              f"  [{min(o['t'] for o in ops) - step['span'][0]:7.0f}..{max(o['end'] for o in ops) - step['span'][0]:7.0f} us]")
    print("  NOTE: phases overlap in wall time; target is the big graph")


# ------------------------------------------------------------------ streams
def print_streams(rows, mains):
    print("\n--- streams (ms/step, union; overlaps another stream = hidden)")
    per = collections.defaultdict(list)
    names = collections.defaultdict(collections.Counter)
    for step in rows:
        by_stream = collections.defaultdict(list)
        for ops in step["phases"].values():
            for o in ops:
                by_stream[o["args"]["stream"]].append(o)
        allb = [o for ops in step["phases"].values() for o in ops]
        other_busy = collections.defaultdict(list)
        for o in allb:
            for s2 in by_stream:
                if s2 != o["args"]["stream"]:
                    other_busy[s2].append(o)
        for s, ops in by_stream.items():
            per[s].append((A.union(ops) / US, A.union(other_busy[s]) / US))
            names[s].update(short(o["name"], 55) for o in ops)
    for s, vals in sorted(per.items(), key=lambda kv: -st.mean(v[0] for v in kv[1])):
        busy = st.mean(v[0] for v in vals)
        other = st.mean(v[1] for v in vals)
        top = ", ".join(f"{n.split('::')[-1][:40]} x{c}" for n, c in names[s].most_common(3))
        print(f"  stream {s:>5}: busy {busy:6.3f} ms/step  exposed(if no other busy) "
              f"<= {max(0.0, busy - other):6.3f}  | {top[:120]}")


# ---------------------------------------------------------------------- SOL
W13_MB, W2_MB = 14.16, 7.08          # int4 g32 per expert, weights+scales
HBM, C2C, TF = 3_640.0, 421.0, 630.0  # GB/s, GB/s, TFLOPS bf16 (worklog roofs)
DECODE_GEMM_MIB = {"o_proj": 48, "qkv_a": 31, "q_b": 16, "wq_b": 16}


def decode_gemm_class(step, o):
    """Position in the layer separates o_proj / pre-attn / indexer wq_b."""
    flash = [x for x in step["phases"]["target"]
             if "flash_fwd_splitkv_mla_fp8_sparse" in x["name"]]
    ts = [f["t"] for f in flash]
    import bisect
    k = bisect.bisect_right(ts, o["t"]) - 1
    k = max(k, 0)
    anchor = k in {0, 1, 2} or (k >= 6 and (k - 2) % 4 == 0)
    before = o["t"] < ts[k]
    if before and anchor and o["t"] > ts[k] - 150:
        return "wq_b (indexer)"
    return "pre-attn (qkv_a/q_b)" if before else "o_proj"


def print_sol(rows, all_ranks):
    print("\n--- cross-rank collectives (per call, us; wire ~= cross-rank min)")
    # (name, ordinal within step) -> {rank: duration}
    tables = collections.defaultdict(dict)
    for rank, rws in all_ranks.items():
        counter = collections.Counter()
        for step in rws:
            for o in sorted(step["phases"]["target"], key=lambda e: e["t"]):
                key = ("AR lamport" if "allreduce_fusion" in o["name"]
                       else "one_shot gather" if "gather_cat" in o["name"]
                       else "one_shot lse_rs" if "lse_reduce_scatter" in o["name"] else None)
                if key:
                    tables[(key, counter[key])][rank] = o["end"] - o["t"]
                    counter[key] += 1
        counter.clear()
    for key in ("AR lamport", "one_shot gather", "one_shot lse_rs"):
        cells = [per for (k, _), per in tables.items() if k == key and len(per) == len(all_ranks)]
        if not cells:
            continue
        n = len(cells) / len(rows)
        means = [st.mean(per.values()) for per in cells]
        spreads = [max(per.values()) - min(per.values()) for per in cells]
        waits = [max(per.values()) - min(per.values()) for per in cells]
        print(f"  {key:18} {n:5.1f}/step  call mean {st.mean(means):5.1f}  cross-rank spread "
              f"{st.mean(spreads):5.1f}  -> total {st.mean(means) * n / US:5.2f} ms/step, "
              f"wait {st.mean(waits) * n / US:5.2f}")

    print("\n--- kernel SOL (mean us per call; bytes model where established)")
    moe_w13, moe_w2, flash, combine = [], [], [], []
    dg_class = collections.defaultdict(list)  # class -> per-call durs
    for step in rows:
        tgt = step["phases"]["target"]
        for o in tgt:
            n = o["name"]
            d = o["end"] - o["t"]
            if "tiered_decode::gemm_kernel<0" in n:
                moe_w13.append(d)
            elif "tiered_decode::gemm_kernel<1" in n:
                moe_w2.append(d)
            elif "flash_fwd_splitkv_mla_fp8_sparse" in n:
                flash.append(d)
            elif "flash_fwd_mla_combine" in n:
                combine.append(d)
            elif "decode_gemm_kernel" in n:
                dg_class[decode_gemm_class(step, o)].append(d)
    def line(label, n, mean_us, bytes_mb=None, floor="hbm", flops_g=None, note=""):
        eff = ""
        if bytes_mb and floor == "hbm":
            eff = f"{bytes_mb / mean_us * 978:7.0f} GB/s {100 * bytes_mb / mean_us / 3.64:4.0f}% HBM"
        if flops_g:
            eff += f" {flops_g / mean_us * 978 / 1e3:5.1f} TF"
        print(f"  {label:32} n/step {n:5.1f}  mean {mean_us:7.2f} us   {eff}  {note}")
    print("  MoE one-kernel (per layer, 75 layers):")
    # implied cold experts from the C2C roof: t = max(hot_floor, c*33.6) + fixed
    cold13 = [(t - 8.0) / 33.6 for t in moe_w13 if t > 41]
    cold2 = [(t - 4.0) / 16.8 for t in moe_w2 if t > 20]
    cold = [max(a, 0) for a in cold13]
    print(f"    w13  mean {st.mean(moe_w13):6.1f} us p10 {q(moe_w13, .1):5.1f} p90 {q(moe_w13, .9):6.1f}"
          f"   -> implied cold/layer {st.mean(cold):.2f} (model t=8+33.6c)")
    print(f"    w2   mean {st.mean(moe_w2):6.1f} us p10 {q(moe_w2, .1):5.1f} p90 {q(moe_w2, .9):6.1f}"
          f"   -> implied cold/layer {st.mean([max(0,(t-4.0)/16.8) for t in moe_w2]):.2f} (model t=4+16.8c)")
    print(f"    FlashMLA main mean {st.mean(flash):5.1f} us  (fixed-cost kernel; sel~512 keys x 656 B "
          f"= {512 * 656 / 1e6:.2f} MB/layer -> bytes are NOT the limit)")
    print(f"    split combine   mean {st.mean(combine):5.1f} us")
    print("  dense GEMM decode_gemm (per call, weights streamed at M=8):")
    for k2, vals in sorted(dg_class.items(), key=lambda kv: -st.mean(kv[1])):
        mib = {"o_proj": 48, "pre-attn (qkv_a/q_b)": 31 or 16}.get(k2)
        note = ""
        if k2 == "o_proj":
            mib, note = 48, "48 MiB -> floor 13.8 us"
        elif k2 == "pre-attn (qkv_a/q_b)":
            note, mib = "split by shape below", None
        elif k2 == "wq_b?":
            note = "indexer wq_b, 16 MiB -> floor 4.6 us (sanity-check label)"
        mb_us = mib * 1.0486 / st.mean(vals) if mib else 0.0  # MB/us == TB/s
        eff = f"{mb_us * 1000:6.0f} GB/s {mb_us / 3.64 * 100:3.0f}% HBM" if mib else ""
        print(f"    {k2:26} n={len(vals) / len(rows):5.1f}/step  mean {st.mean(vals):6.2f} us  {eff}  {note}")
        if k2 == "pre-attn (qkv_a/q_b)":
            h = collections.Counter(round(v) for v in vals)
            top = ", ".join(f"{v}us x{c}" for v, c in h.most_common(6))
            print(f"      duration modes: {top}")
    print("  remaining GEMM/glue kernels (top by total ms/step):")
    perstep = collections.defaultdict(list)
    for step in rows:
        cnt = collections.Counter()
        for o in step["phases"]["target"]:
            n = short(o["name"], 60)
            if ("nvjet" in o["name"] or "splitKreduce" in o["name"] or "cutlass_kernel" in o["name"] or
                    "deep_gemm" in o["name"] or "triton_" in o["name"] or "mqa_logits" in o["name"] or
                    "concat_and_cache" in o["name"] or "indexer" in o["name"] or "grouped_topk" in o["name"] or
                    "route_prep" in o["name"] or "act_kernel" in o["name"] or "finalize" in o["name"]):
                cnt[n] += o["end"] - o["t"]
        for n, v in cnt.items():
            perstep[n].append(v)
    for n, vals in sorted(perstep.items(), key=lambda kv: -st.mean(kv[1]))[:18]:
        print(f"    {n[:58]:58} {st.mean(vals) / US:6.3f} ms/step")


def main():
    d = Path(sys.argv[1])
    ranks = {}
    for path in sorted(d.glob("*.pt.trace.json.gz")):
        rank = int(A.RANK_RE.search(path.name).group(1))
        ranks[rank] = A.steps(A.load(path))
        print(f"rank {rank}: {len(ranks[rank])} steps", file=sys.stderr)
    n = min(len(r) for r in ranks.values())
    ranks = {r: rows[:n] for r, rows in ranks.items()}
    rows = ranks[0]
    periods = [r["period"] for r in rows]
    print(f"== {d.name}: {len(rows)} steps, period mean {st.mean(periods) / US:.2f} "
          f"p50 {st.median(periods) / US:.2f} ms")
    phase = collections.defaultdict(list)
    idle = []
    for step in rows:
        everything = [o for ops in step["phases"].values() for o in ops]
        busy = A.union(everything) / US
        idle.append(step["period"] / US - busy)
        for ph, ops in step["phases"].items():
            phase[ph].append(A.union(ops) / US)
    for ph in ("target", "logits", "draft", "host-side"):
        print(f"  {ph:10} busy {st.mean(phase[ph]):6.3f} ms/step")
    print(f"  GPU idle (period - any-busy) {st.mean(idle):.3f} ms/step")
    print_streams(rows, None)
    if "--seq" in sys.argv:
        step = med_step(rows)
        phase_skeleton(step)
        print_seq(step, 39, "skip layer (per c2k3: ~245 us)")
        print_seq(step, 42, "anchor layer (indexer)")
        print_seq(step, 77, "last layer -> next step")
    if "--sol" in sys.argv:
        print_sol(rows, ranks)


if __name__ == "__main__":
    main()
