# contiguous runs of SASS by exe count: start-end offsets, exe, samples share
import csv,sys
rows=list(csv.reader(open(sys.argv[1]))); h=rows[1]; data=rows[2:]
S=h.index("Warp Stall Sampling (All Samples)"); E=h.index("Instructions Executed")
base=int(data[0][0],16)
tot=sum(int(r[S] or 0) for r in data)
runs=[]
for r in data:
    a=int(r[0],16)-base; e=int(r[E] or 0); s=int(r[S] or 0)
    if runs and runs[-1][2]==e: runs[-1][1]=a; runs[-1][3]+=s; runs[-1][4]+=1; runs[-1][5].append(r[1].strip().split()[0] if not r[1].strip().startswith('@') else r[1].strip().split()[1])
    else: runs.append([a,a,e,s,1,[r[1].strip().split()[0]]])
for a,b,e,s,n,ops in runs:
    if s/tot>=0.004 or n>=40:
        print(f"{a:05x}-{b:05x} n={n:4d} exe={e:7d} {100*s/tot:5.1f}%  {' '.join(o.split('.')[0] for o in ops[:8])}")
