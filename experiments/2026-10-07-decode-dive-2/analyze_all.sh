#!/usr/bin/env bash
# All decode analyses on a decode-mem-dive/run.sh profile set:
#   analyze_all.sh <timing tag> <nsys tag> <nsyslong tag>
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments
R=/e/fscratch/profound/${USER}/decode-mem-dive
T=${R}/$1/trace
PY=.venv/bin/python
for w in window-0 window-10 window-11; do
  [[ -d ${T}/${w} ]] || continue
  echo "################ ${w}"
  ${PY} ${X}/2026-09-26-mimo-decode-profile/analyze.py ${T}/${w} 2>&1 | tail -24
  ${PY} ${X}/2026-10-07-decode-mem-dive/coll_wait.py ${T}/${w}
  ${PY} ${X}/2026-10-07-decode-mem-dive/moe_skew.py ${T}/${w}
done
echo "################ verify graph (window-0)"
${PY} ${X}/2026-09-28-agentic-decode-bench/verify_graph_dive.py ${T}/window-0 2>&1 | sed -n '1,75p'
echo "################ layer timeline (window-0, rank 0)"
${PY} ${X}/2026-10-07-decode-mem-dive/layer_timeline.py ${T}/window-0/*rank0.*.gz --layers 40,41,42
echo "################ segments"
${PY} ${X}/2026-10-07-dense-l2-prefetch/segments.py l2pf0-timing dg1b-timing $1
NS=/e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys
for t in $2 $3; do
  echo "################ nsys ${t}"
  [[ -f ${R}/${t}/nsys.sqlite ]] || ${NS} export --type sqlite --force-overwrite true -o ${R}/${t}/nsys.sqlite ${R}/${t}/nsys.nsys-rep >/dev/null 2>&1
  ${PY} ${X}/2026-09-29-glm-verify-kernels/nsys_steps.py ${R}/${t}/nsys.sqlite 2>&1 | head -7
done
