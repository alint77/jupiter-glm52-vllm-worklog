"""Where is the all-reduce skew absorbed: post-attention or post-MoE?"""
import collections, statistics as stat
from tracelib import prep_depth, steps, gpu_ops

depth = 3
buckets = collections.defaultdict(list)
nsteps = 0
for rank in range(4):
    ev = prep_depth(rank, depth)
    for a, b in steps(ev):
        ops = sorted(gpu_ops(ev, a, b), key=lambda e: e["t"])
        nsteps += 1
        for i, e in enumerate(ops):
            if "cross_device_reduce" not in e["name"]:
                continue
            # look back for the nearest identifying producer on any stream
            tag = "unknown"
            for j in range(i - 1, max(-1, i - 40), -1):
                n = ops[j]["name"]
                if "moe_sum_vec" in n or "CUDAFunctor_add<c10::BFloat16>" in n:
                    tag = "post-MoE"; break
                if "flash_fwd_mla_combine" in n or "_correct_attn_cp_out" in n or "ReduceScatter" in n:
                    tag = "post-attention"; break
                if "cutlass::device_kernel" in n and tag == "unknown":
                    continue
            buckets[tag].append(e["dur"])

print(f"=== all-reduce by position, {nsteps} rank-steps ===")
print(f"{'position':18s} {'n/step':>8} {'ms/step':>9} {'mean us':>9} {'p50':>8} {'p90':>8} {'max':>9}")
tot = 0
for tag, d in sorted(buckets.items(), key=lambda x: -sum(x[1])):
    d = sorted(d); n = len(d)
    tot += sum(d)
    print(f"{tag:18s} {n/nsteps:8.1f} {sum(d)/1000/nsteps:9.3f} {sum(d)/n:9.2f} "
          f"{d[n//2]:8.2f} {d[int(.9*n)]:8.2f} {d[-1]:9.2f}")
print(f"{'TOTAL':18s} {sum(len(v) for v in buckets.values())/nsteps:8.1f} {tot/1000/nsteps:9.3f}")

# excess over the per-position floor
print("\nwait above per-position p5 floor:")
for tag, d in buckets.items():
    d = sorted(d); floor = d[int(.05 * (len(d) - 1))]
    print(f"  {tag:18s} floor={floor:6.2f} us   wait={(sum(d)-floor*len(d))/1000/nsteps:6.3f} ms/step")
