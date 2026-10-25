#!/usr/bin/env bash
# shard_probe.py on cold shards: 1 process and 4 processes (ranks 0-3, same
# shard, as the 4 workers), mmap vs eager, a different shard per case.
cd /e/project1/profound/alint77/vllm
M=/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16
P="numactl --cpunodebind=0 --membind=0 .venv/bin/python agent_space/experiments/2026-10-25-boot-time/shard_probe.py"
sh() { printf "%s/model-%05d-of-00282.safetensors" ${M} $1; }
fincore -b $(sh 30) $(sh 31) $(sh 32) $(sh 33) $(sh 34) $(sh 35) $(sh 36) $(sh 37)
$P mmap 0 $(sh 30)
$P eager 0 $(sh 31)
$P mmap 0 $(sh 32)
$P eager 0 $(sh 33)
for r in 0 1 2 3; do numactl --cpunodebind=$r --membind=$r .venv/bin/python agent_space/experiments/2026-10-25-boot-time/shard_probe.py mmap $r $(sh 34) & done; wait
for r in 0 1 2 3; do numactl --cpunodebind=$r --membind=$r .venv/bin/python agent_space/experiments/2026-10-25-boot-time/shard_probe.py eager $r $(sh 35) & done; wait
echo "--- warm repeat (page cache / pagepool)"
$P mmap 0 $(sh 30)
$P eager 0 $(sh 31)
