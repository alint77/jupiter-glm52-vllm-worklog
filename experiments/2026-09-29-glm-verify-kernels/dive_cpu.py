import sys,collections
from pathlib import Path
sys.path.insert(0,'../2026-09-26-mimo-decode-profile'); import analyze as A
w=Path(sys.argv[1]); steps_want=[int(x) for x in sys.argv[2].split(',')]
f=sorted(w.glob('*rank0*.pt.trace.json.gz'))[0]
ev=A.load(f)
launch={e['args'].get('correlation'):e for e in ev if e.get('cat') in A.LAUNCH_CATS}
rows=A.steps(ev)
cpu=[e for e in ev if e.get('cat') in ('python_function','cpu_op','user_annotation','cuda_runtime','cuda_driver')]
for s in steps_want:
    row=rows[s]; tgt=sorted(row['phases']['target'],key=lambda o:o['t'])
    gl=launch[tgt[0]['args']['correlation']]
    prev=rows[s-1]; pend=max(o['end'] for ops in prev['phases'].values() for o in ops)
    lo=pend-0.5; hi=tgt[0]['t']
    print(f'== step {s}: prev GPU end {0:.3f}, graph launch call at {gl["t"]-pend:.3f} (dur {gl["end"]-gl["t"]:.3f}), GPU entry {tgt[0]["t"]-pend:.3f} ms')
    # GPU kernels on rank0 between prev end and entry
    gk=[e for e in ev if e.get('cat') in A.GPU_CATS and pend-0.01<e['t']<hi]
    c=collections.Counter(); d=collections.defaultdict(float)
    for e in gk: c[e['name'][:60]]+=1; d[e['name'][:60]]+=e['end']-e['t']
    print('  GPU ops before entry:', [(k,c[k],round(d[k],3)) for k in sorted(d,key=lambda k:-d[k])[:6]])
    top=[e for e in cpu if e['t']<hi and e['end']>lo and e['end']-e['t']>0.15]
    top.sort(key=lambda e:e['t'])
    for e in top[:40]:
        print(f'  {e["t"]-pend:8.3f} {e["end"]-e["t"]:7.3f} {e.get("cat"):16s} {e["name"][:110]}')
