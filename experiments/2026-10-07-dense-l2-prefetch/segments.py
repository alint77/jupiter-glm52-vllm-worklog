"""Per-layer critical-path segments (rank 0, MoE layers 3-76), median us, for
several profiled runs (decode-mem-dive/run.sh timing): attention up to o_proj,
o_proj, the attention AR, router/top-k up to route_prep, the MoE chain, the
MoE AR (incl. waiting) and the pre-attention part up to the next FlashMLA, plus
the kernels between the split combine and o_proj.

    segments.py <run tag>...   (dirs under /e/fscratch/.../decode-mem-dive)
"""
import collections
import glob
import gzip
import json
import statistics as st
import sys

ROOT = "/e/fscratch/profound/naeimitabiei1/decode-mem-dive"


def run(tag):
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{ROOT}/{tag}/trace/window-*/*rank0.*.gz")):
        ev = json.load(gzip.open(f))["traceEvents"]
        k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        mla = [i for i, e in enumerate(k) if "flash_fwd_splitkv_mla_fp8_sparse" in e["name"]]
        steps, cur = [], []
        for i in mla:
            if cur and k[i]["ts"] - k[cur[-1]]["ts"] > 1000:
                steps.append(cur)
                cur = []
            cur.append(i)
        for s in (s for s in steps if len(s) == 78):
            for L in range(3, 77):
                ks = k[s[L]:s[L + 1]]
                t0 = ks[0]["ts"]

                def first(pat, after=0):
                    return next((e for e in ks if pat in e["name"] and e["ts"] >= after), None)

                ar1, prep, fin = first("allreduce_fusion"), first("route_prep"), first("finalize_kernel")
                if not (ar1 and prep and fin):
                    continue
                # o_proj: the last GEMM (cuBLAS or decode_gemm) before attention's AR
                gemms = [e for e in ks if e["ts"] < ar1["ts"]
                         and (e["name"].startswith("nvjet") or "decode_gemm" in e["name"])]
                op = gemms[-1] if gemms else None
                if not op:
                    continue
                ar2 = first("allreduce_fusion", fin["ts"])
                if not ar2:
                    continue
                uv = [e for e in gemms[:-1] if e["name"].startswith("nvjet")]
                lse = first("lse_reduce_scatter")
                end = lambda e: e["ts"] + e["dur"]  # noqa: E731
                out["MLA start -> o_proj start"].append(op["ts"] - t0)
                out["  DCP LSE reduce-scatter"].append(lse["dur"] if lse else float("nan"))
                out["  W_UV"].append(uv[-1]["dur"] if uv else float("nan"))
                out["o_proj"].append(op["dur"])
                out["o_proj end -> attn AR end"].append(end(ar1) - end(op))
                out["attn AR end -> route_prep"].append(prep["ts"] - end(ar1))
                out["MoE chain"].append(end(fin) - prep["ts"])
                out["MoE AR incl. wait"].append(ar2["dur"])
                out["MoE AR end -> next FlashMLA"].append(k[s[L + 1]]["ts"] - end(ar2))
                out["layer"].append(k[s[L + 1]]["ts"] - t0)
    return out


tags = sys.argv[1:]
res = [run(t) for t in tags]
print(f"{'segment (median us)':32s}" + "".join(f"{t:>16s}" for t in tags))
for key in res[0]:
    print(f"{key:32s}" + "".join(f"{st.median(r[key]) if r.get(key) else float('nan'):16.1f}" for r in res))
print(f"{'layers':32s}" + "".join(f"{len(r['layer']):16d}" for r in res))
