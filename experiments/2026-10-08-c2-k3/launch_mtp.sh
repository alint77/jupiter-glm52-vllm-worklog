#!/usr/bin/env bash
# One fresh hold per arm: MTP server (this dir's serve.sh), probe with the
# profiled windows first, then the trace analyses; releases the node.
#   launch_mtp.sh <tag> <spec_k> <max_num_seqs> <reserve> <capture sizes> [probe args]
cd /e/project1/profound/alint77/vllm
C=agent_space/experiments/2026-10-08-c2-k3
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
X=agent_space/experiments
tag=$1 k=$2 seqs=$3 r=$4 caps=$5; shift 5
j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch")
echo "${j} ${tag}"
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "SPEC=mtp SPEC_K=${k} MAX_NUM_SEQS=${seqs} RESERVE_GB=${r} CAPTURE_SIZES=${caps} COMPILE_SIZES= ${C}/run.sh ${tag} --with-profile $*" > ${C}/run-${tag}.log 2>&1
scancel "${j}"
T=/e/fscratch/profound/${USER}/c2-k3/${tag}/trace
for w in ${T}/*/; do
  echo "######## ${w}"
  .venv/bin/python ${X}/2026-09-28-agentic-decode-bench/kernel_tally.py ${w} --top 30
  .venv/bin/python ${X}/2026-10-07-decode-mem-dive/coll_wait.py ${w}
  .venv/bin/python ${X}/2026-10-07-decode-mem-dive/layer_timeline.py ${w}/*rank0.*.gz --layers 40
done > ${C}/analysis-${tag}.txt 2>&1
echo done > ${C}/analysis-${tag}.done
