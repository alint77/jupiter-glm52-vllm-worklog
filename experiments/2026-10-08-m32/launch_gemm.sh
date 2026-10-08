#!/usr/bin/env bash
# decode_gemm vs cuBLAS sweep (bench_gemm_m32.py) on a fresh hold, GPU 0
# NUMA-bound; two runs.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=00:45:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "for r in 1 2; do CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 .venv/bin/python ${D}/bench_gemm_m32.py | sed \"s/}\$/, \\\"rep\\\": \$r}/\"; done" > ${D}/gemm-${j}.jsonl 2> ${D}/gemm-${j}.err
scancel "${j}"
echo ${j} > ${D}/gemm.done
