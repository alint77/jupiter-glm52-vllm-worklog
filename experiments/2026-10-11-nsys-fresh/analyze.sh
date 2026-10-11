#!/usr/bin/env bash
# Decode tables for the nfr-c<c> runs with the nsys-sol tools (an/: bd-*, kc-*,
# st-*, sol-w*.md, bd-w*.md); the sizes/mixed reports get kc-* and sizes_table.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-nsys-fresh
S=agent_space/experiments/2026-10-11-nsys-sol
T=agent_space/experiments/2026-10-09-tp-sliced-experts
P=.venv/bin/python
R=/e/fscratch/profound/${USER}/sliced-prof
mkdir -p ${D}/an
NS=/e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys
for f in ${R}/nfr-c*/nsys.*.nsys-rep; do [[ -f ${f%.nsys-rep}.sqlite ]] || ${NS} export --type sqlite -o ${f%.nsys-rep}.sqlite ${f} >/dev/null 2>&1 & done; wait
runs=""
for c in 1 2 4 8 16; do
  r=$(ls -d ${R}/nfr-c${c}-* | tail -1 | xargs basename); runs+="${runs:+,}${c}:${r}"
  for w in 1 2; do for d in 0 1 2 3; do
    $P ${T}/nsys_bd.py ${R}/${r}/nsys.${w}.nsys-rep --device ${d} --json ${D}/an/bd-c${c}-w${w}-d${d}.json > ${D}/an/bd-c${c}-w${w}-d${d}.txt 2>&1 &
  done; done
done
wait
for c in 1 2 4 8 16; do r=$(ls -d ${R}/nfr-c${c}-* | tail -1); for w in 1 2; do for d in 0 1 2 3; do
  $P ${T}/kchain_nsys.py ${r}/nsys.${w}.sqlite --device ${d} > ${D}/an/kc-c${c}-w${w}-d${d}.txt 2>&1 &
  $P ${S}/staging_cost.py ${r}/nsys.${w}.sqlite --device ${d} > ${D}/an/st-c${c}-w${w}-d${d}.txt 2>&1 &
done; done; done
wait
for w in 1 2; do
  SOL_RUNS=${runs} SOL_AN=${D}/an $P ${S}/sol.py ${w} --md > ${D}/an/sol-w${w}.md
  SOL_AN=${D}/an $P ${S}/bd_table.py ${w} > ${D}/an/bd-w${w}.md
done
for what in sizes mixed; do r=$(ls -d ${R}/nfr-${what}-* | tail -1)
  for f in ${r}/nsys.*.nsys-rep; do q=${f%.nsys-rep}.sqlite
    [[ -f ${q} ]] || /e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys export --type sqlite -o ${q} ${f} >/dev/null 2>&1 &
  done; wait
  $P agent_space/experiments/2026-10-11-prefill-sizes/sizes_table.py ${r} > ${D}/an/${what}-table.md 2>&1
done
echo analyzed
