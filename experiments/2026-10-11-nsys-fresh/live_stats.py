import re, sys, collections
E = re.compile(r"INFO (\d+-\d+ [\d:]+) \[loggers.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
S = re.compile(r"SpecDecoding metrics: .*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+?), Avg")
for path in sys.argv[1:]:
    rows, acc, dra, pos = [], 0, 0, []
    for l in open(path, errors="replace"):
        m = E.search(l)
        if m:
            rows.append((m[1], *map(float, m.groups()[1:])))
        m = S.search(l)
        if m:
            a, d = int(m[1]), int(m[2]); acc += a; dra += d
            pos.append(([float(x) for x in m[3].split(",")], d // 3))
    act = [r for r in rows if r[3] > 0 or r[1] > 0 or r[2] > 0]
    ptok = sum(r[1] for r in rows) * 10; gtok = sum(r[2] for r in rows) * 10
    dec = [r for r in act if r[1] == 0 and r[3] > 0]
    by_run = collections.defaultdict(list)
    for r in dec: by_run[int(r[3])].append(r[2] / r[3])
    drafts = dra / 3
    pp = [sum(p[i] * w for p, w in pos) / max(1, sum(w for _, w in pos)) for i in range(3)]
    print(f"== {path.split('-')[-1]}  {rows[0][0]} .. {rows[-1][0]}")
    print(f"  logged windows {len(rows)}, active {len(act)} ({len(act)*10/60:.1f} min)")
    print(f"  prompt tokens computed ~{ptok/1e6:.2f}M, generated ~{gtok/1e3:.0f}K")
    print(f"  running dist (active): {dict(sorted(collections.Counter(int(r[3]) for r in act).items()))}, max waiting {max(r[4] for r in rows):.0f}, windows with waiting>0: {sum(r[4]>0 for r in rows)}")
    print(f"  max KV usage {max(r[5] for r in rows):.1f}%, final prefix hit {rows[-1][6]:.1f}%")
    print(f"  pure-decode windows {len(dec)}: per-req gen tok/s by running: " + ", ".join(f"r{k}:{sum(v)/len(v):.0f} (n={len(v)})" for k, v in sorted(by_run.items())))
    if drafts: print(f"  spec: mean accept len {1+acc/drafts:.2f}, per-pos {', '.join(f'{x:.3f}' for x in pp)}, drafts {drafts:.0f}")
