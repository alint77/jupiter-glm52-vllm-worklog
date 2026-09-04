import json, os, re, collections
m = f"/e/fscratch/profound/{os.popen('id -un').read().strip()}/models/GLM-5.3-W4A16"
idx = json.load(open(f"{m}/model.safetensors.index.json"))["weight_map"]
# read dtype+shape per tensor from each shard header (cheap: header only)
import struct
shards = collections.defaultdict(list)
for name, f in idx.items(): shards[f].append(name)
DT = {"BF16":2,"F16":2,"F32":4,"F8_E4M3":1,"I32":4,"U8":1,"I8":1,"I64":8,"F64":8,"BOOL":1}
buckets = collections.Counter()
for fn, names in shards.items():
    with open(os.path.join(m, fn), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name in names:
        meta = hdr.get(name)
        if not meta: continue
        nel = 1
        for d in meta["shape"]: nel *= d
        b = nel * DT.get(meta["dtype"], 2)
        dt = meta["dtype"]
        if re.search(r"mlp\.experts\.\d+\.", name):  key = "routed experts"
        elif "shared_experts" in name:               key = "shared experts"
        elif "self_attn" in name:                    key = "self_attn"
        elif re.search(r"\.mlp\.(gate|up|down)_proj", name): key = "dense MLP (first 3 layers)"
        elif "embed" in name or "lm_head" in name:   key = "embed/lm_head"
        else:                                        key = "norms/other"
        buckets[(key, "quantized" if dt in ("I32","U8","I8") else dt)] += b
tot = sum(buckets.values())
print(f"checkpoint total {tot/2**30:8.2f} GiB")
agg = collections.Counter()
for (k, dt), b in buckets.items(): agg[k] += b
for k, b in agg.most_common():
    dts = {dt: v for (kk, dt), v in buckets.items() if kk == k}
    print(f"  {k:28s} {b/2**30:8.2f} GiB  {100*b/tot:5.1f}%   "
          + ", ".join(f"{dt} {v/2**30:.2f}" for dt, v in sorted(dts.items(), key=lambda i:-i[1])))
bf16_nonexpert = sum(b for (k, dt), b in buckets.items()
                     if dt in ("BF16","F16") and k in ("self_attn","shared experts","dense MLP (first 3 layers)"))
print(f"\nbf16 weights read EVERY step (self_attn + shared experts + dense MLP):")
print(f"  {bf16_nonexpert/2**30:.2f} GiB total, {bf16_nonexpert/4/2**30:.2f} GiB per GPU at TP4")
print(f"  at ~4 TB/s HBM that is {bf16_nonexpert/4/4e12*1000:.2f} ms/step of pure weight streaming")
print(f"  measured dense/shared GEMM bucket: 5.734 ms/step")
