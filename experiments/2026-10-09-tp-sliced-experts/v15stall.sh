#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
./ncu1.sh v15co td_v15:TD_COMPUTE_ONLY 8 38,0 --shared 1 2>&1 | head -12
source /e/project1/profound/alint77/vllm/agent_space/jupiter-env.sh >/dev/null 2>&1
ncu --import logs/ncu-v15co.ncu-rep --page source --csv --print-source sass 2>/dev/null > logs/ncu-v15co-sass2.csv
/e/project1/profound/alint77/vllm/.venv/bin/python - logs/ncu-v15co-sass2.csv <<'PY'
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1]))); h = rows[1]
cols = [c for c in h if c.startswith("stall_") or "Warp Stall Sampling" in c]
isrc, iall = h.index("Source"), h.index("Warp Stall Sampling (All Samples)")
stall_cols = [i for i, c in enumerate(h) if c.startswith("smsp__pcsamp_warps_issue_stalled_") and not c.endswith("_not_issued")]
ins = []
for r in rows[2:]:
    try: ins.append((r[isrc].strip(), int(r[iall] or 0), {h[i]: int(r[i] or 0) for i in stall_cols}))
    except Exception: pass
idx = [i for i, x in enumerate(ins) if "HMMA.16816.F32 " in x[0] or x[0].startswith("HMMA.16816.F32 ")]
blk = set(); cur = [idx[0]] if idx else []
for i in idx[1:] + [10**9]:
    if i != 10**9 and i - cur[-1] <= 80: cur.append(i); continue
    if len(cur) >= 8: blk.update(range(cur[0] - 30, cur[-1] + 6))
    cur = [i]
tot = sum(x[1] for x in ins); inb = sum(ins[i][1] for i in blk if i < len(ins))
print(f"samples {tot}, in routed MMA blocks {inb} ({inb/max(tot,1):.0%})")
agg = collections.Counter(); op = collections.Counter()
for i in blk:
    if i >= len(ins): continue
    s, a, st = ins[i]
    for k, v in st.items(): agg[k.replace("smsp__pcsamp_warps_issue_stalled_", "")] += v
    o = s.split()[1] if s.startswith("@") else s.split()[0]
    op[o.split(".")[0]] += a
t = sum(agg.values()) or 1
print("block stall reasons:", ", ".join(f"{k}:{v/t:.0%}" for k, v in agg.most_common(8)))
print("block samples by opcode:", ", ".join(f"{k}:{v/max(inb,1):.0%}" for k, v in op.most_common(8)))
PY
