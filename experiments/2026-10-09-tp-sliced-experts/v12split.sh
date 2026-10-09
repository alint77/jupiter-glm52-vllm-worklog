#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
./ab.sh s9 8 '38,0 38,4' 'td_v13:|--shared 1' 'td_v13:TD_COMPUTE_ONLY|--shared 1'
./ncu1.sh v13m8 td_v13: 8 38,0 --shared 1 > /dev/null 2>&1
/e/project1/profound/alint77/vllm/.venv/bin/python - logs/ncu-v13m8-sass.csv <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]))); h = rows[1]
isrc, iexe = h.index('Source'), h.index('Instructions Executed')
ins = []
for r in rows[2:]:
    try: ins.append((r[isrc].strip(), int(r[iexe] or 0)))
    except Exception: pass
tot = sum(e for _, e in ins)
idx = [i for i, (s, _) in enumerate(ins) if 'HMMA' in s]
blk, cur = set(), [idx[0]]
for i in idx[1:] + [10**9]:
    if i != 10**9 and i - cur[-1] <= 80: cur.append(i); continue
    if len(cur) >= 8: blk.update(range(cur[0] - 30, cur[-1] + 6))
    cur = [i]
inb = sum(ins[i][1] for i in blk if i < len(ins))
print(f'executed {tot/1e6:.2f} M warp-instr; MMA blocks {inb/1e6:.2f} M ({inb/tot:.0%})')
PY
source /e/project1/profound/alint77/vllm/agent_space/jupiter-env.sh >/dev/null 2>&1
ncu --import logs/ncu-v13m8.ncu-rep --page raw 2>/dev/null | grep -E ' gpu__time_duration.sum | smsp__issue_active.avg.pct_of_peak_sustained_active | dram__throughput.avg.pct_of_peak_sustained_elapsed ' | awk '{print $1, $NF}'
