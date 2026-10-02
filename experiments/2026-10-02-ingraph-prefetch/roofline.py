"""Roofline of the prefill MoE Marlin kernels from a trace.
roofline.py <trace> <tokens> [peak_tflops=630] [peak_tbps=3.6]"""
import gzip, json, sys, statistics as st, collections

H, I, TOPK, E_GLOBAL, RANKS = 6144, 2048, 8, 256, 4
W13 = 2 * I * H // 2 + 2 * I * H // 32 * 2      # int4 weights + bf16 group-32 scales
W2 = H * I // 2 + H * I // 32 * 2
EXPERT = W13 + W2
F13, F2 = 2 * H * 2 * I, 2 * I * H                # FLOPs per routed token
tr, tokens = sys.argv[1], int(sys.argv[2])
PF = float(sys.argv[3]) * 1e12 if len(sys.argv) > 3 else 630e12
PB = float(sys.argv[4]) * 1e12 if len(sys.argv) > 4 else 3.6e12
end = lambda e: e["ts"] + e["dur"]
ev = json.load(gzip.open(tr))["traceEvents"]
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"
       and e["name"].startswith(f"execute_context_1({tokens})")]
W0 = min(e["ts"] for e in ann); W1 = max(end(e) for e in ann)
k = sorted((e for e in ev if e.get("cat") == "kernel" and W0 <= e["ts"] < W1), key=lambda e: e["ts"])
cold = sorted((e for e in ev if e.get("cat") == "gpu_memcpy" and W0 - 5000 <= e["ts"] < W1
               and "HtoD" in e["name"] and e["args"].get("bytes", 0) >= 5 << 20), key=lambda e: e["ts"])
mar = [e for e in k if "marlin" in e["name"]]
layers = [mar[i:i + 4] for i in range(0, len(mar), 4)]
# Copy bursts -> bytes per staged layer, in order (the dense-layer copy is layer 0's).
bursts = []
for c in cold:
    if bursts and c["ts"] - bursts[-1][1] < 20: bursts[-1][1] = end(c); bursts[-1][2] += c["args"]["bytes"]
    else: bursts.append([c["ts"], end(c), c["args"]["bytes"]])
ncold = [round(b[2] / EXPERT) for b in bursts][:len(layers)]

routes = tokens * TOPK / RANKS                        # per rank, if balanced
for bm in (8, 16, 32, 48, 64):
    if tokens * TOPK / E_GLOBAL / bm < 0.9: break
per_exp = tokens * TOPK / E_GLOBAL
rows = max(bm, -(-per_exp // bm) * bm)                # padded rows per expert (mean-load)

span = [end(l[-1]) - l[0]["ts"] for l in layers]
streams = collections.Counter(e["args"]["stream"] for e in mar)
print(f"{tokens} tokens: {len(layers)} MoE layers, block_size_m {bm}, ~{per_exp:.0f} tokens/expert "
      f"-> {rows:.0f} padded rows/expert")
n_exp = E_GLOBAL // RANKS
B = n_exp * EXPERT
Fu = routes * (F13 + F2)
Fp = n_exp * rows * (F13 + F2)
t_mem, t_fu, t_fp = B / PB, Fu / PF, Fp / PF
s = st.median(span) * 1e-6
print(f"per rank per layer: {n_exp} experts, weights {B/1e9:.3f} GB, useful {Fu/1e9:.1f} GFLOP, padded {Fp/1e9:.1f} GFLOP")
print(f"  intensity useful {Fu/B:.0f} F/B, padded {Fp/B:.0f} F/B, ridge {PF/PB:.0f} F/B")
print(f"  SOL: memory {t_mem*1e6:.0f} us | compute useful {t_fu*1e6:.0f} us, padded {t_fp*1e6:.0f} us")
print(f"  + cold copy writes into HBM during the MoE (median {st.median(ncold)} experts, "
      f"{st.median(ncold)*EXPERT/1e9:.2f} GB): memory SOL {(B+st.median(ncold)*EXPERT)/PB*1e6:.0f} us")
print(f"  measured Marlin span (first start..last end) median {s*1e6:.0f} us, min {min(span):.0f}, max {max(span):.0f}")
print(f"  => {B/s/1e12:.2f} TB/s ({100*B/s/PB:.0f}% of HBM SOL), {Fu/s/1e12:.0f} TFLOPS useful "
      f"({100*Fu/s/PF:.0f}%), {Fp/s/1e12:.0f} TFLOPS padded ({100*Fp/s/PF:.0f}%)")

# Per position within the layer: kernels sorted by start are [A w13, B w13, A w2, B w2].
names = ["A w13", "B w13", "A w2", "B w2"]
for pos in range(4):
    ds = [l[pos]["dur"] for l in layers]
    g = collections.Counter(tuple(l[pos]["args"]["grid"]) for l in layers)
    xs = ncold[:len(layers)]
    c = st.correlation(xs, ds[:len(xs)]) if st.pstdev(xs) > 0 else float("nan")
    print(f"  {names[pos]}: median {st.median(ds):.0f} us, grids {dict(g)}, corr with cold experts {c:+.2f}")
print(f"  cold experts per layer: median {st.median(ncold)}, range {min(ncold)}-{max(ncold)}")
