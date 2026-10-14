import json, os, re, sys, collections
H = sys.argv[1]
DT = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "F8_E8M0": 1}
cls = collections.Counter(); n = collections.Counter(); per_layer = collections.defaultdict(collections.Counter)
dtypes = collections.Counter(); bad = 0
for f in sorted(os.listdir(H)):
    try: h = json.load(open(os.path.join(H, f)))
    except Exception: bad += 1; continue
    for k, v in h.items():
        if k == "__metadata__": continue
        b = v["data_offsets"][1] - v["data_offsets"][0]
        m = re.search(r"layers\.(\d+)\.", k)
        if "vision_tower" in k or "mm_projector" in k: c = "vision"
        elif "embed_tokens" in k: c = "embed"
        elif "lm_head" in k: c = "lm_head"
        elif ".mlp.experts." in k: c = "routed_experts"
        elif "shared_expert" in k: c = "shared_experts"
        elif re.search(r"\.mlp\.(gate|router)", k) and "proj" not in k: c = "router"
        elif ".mlp." in k: c = "moe_other/dense_mlp" if m and int(m[1]) != 0 else "dense_mlp(layer0)"
        elif "self_attn" in k: c = "attn"
        else: c = "other"
        cls[c] += b; n[c] += 1; dtypes[(c, v["dtype"])] += b
        if m: per_layer[int(m[1])][c] += b
print("bad headers", bad)
G = 2**30
for c, b in cls.most_common(): print(f"{c:22s} {b/G:9.2f} GiB  tensors {n[c]}")
print(f"{'TOTAL':22s} {sum(cls.values())/G:9.2f} GiB")
print({f"{k[0]}:{k[1]}": round(v/G, 2) for k, v in dtypes.items()})
L = sorted(per_layer)
print("layers", L[0], "..", L[-1], len(L))
for i in (0, 1, 3, 4):
    print(i, {k: round(v/2**20, 1) for k, v in per_layer[i].items()}, "MiB")
mla = [i for i in L if per_layer[i]["attn"] and i in (3,7,11)] 
