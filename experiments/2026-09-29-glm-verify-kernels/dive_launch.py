import sys,statistics
from pathlib import Path
sys.path.insert(0,'../2026-09-26-mimo-decode-profile'); import analyze as A
for w in sys.argv[1:]:
  for rank in range(4):
    f=sorted(Path(w).glob(f'*rank{rank}.*.pt.trace.json.gz'))[0]
    ev=A.load(f)
    launch={e['args'].get('correlation'):e for e in ev if e.get('cat') in A.LAUNCH_CATS}
    ex=sorted((e for e in ev if e.get('cat')=='user_annotation' and e['name'].startswith('execute_')),key=lambda e:e['t'])
    rows=A.steps(ev); out=[]
    for s,row in enumerate(rows):
        tgt=min(row['phases']['target'],key=lambda o:o['t'])
        gl=launch[tgt['args']['correlation']]
        e0=[e for e in ex if e['t']<=gl['t']][-1]
        out.append((s,gl['t']-e0['t'],gl['end']-gl['t']))
    prep=[o[1] for o in out]; dur=[o[2] for o in out]
    slow=[o[0] for o in out if o[2]>0.5]
    print(f'{Path(w).name} r{rank}: prep p50 {statistics.median(prep):.2f} p90 {sorted(prep)[int(.9*len(prep))]:.2f} | graphLaunch p50 {statistics.median(dur):.3f} p90 {sorted(dur)[int(.9*len(dur))]:.2f} max {max(dur):.2f}  slow(>0.5ms) {len(slow)}/{len(out)} {slow[:20]}')
