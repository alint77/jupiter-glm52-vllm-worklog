# Claude Code AutoRound c4 host

`server.sbatch` serves GLM-5.2 AutoRound W4G64 with MTP3 on one JUPITER
Booster node. It uses TP4/EP4/DCP4, allows four concurrent sequences, captures
verification batches 4/8/12/16, and retains a 7 GB HBM reserve.

From the vLLM checkout:

```bash
./claude-local-c4.sh
```

The c4 launcher has separate Slurm state, credentials, caches, and job name,
so it can coexist with `claude-local.sh`. Its lifecycle commands are:

```bash
./claude-local-c4.sh --start
./claude-local-c4.sh --status
./claude-local-c4.sh --monitor
./claude-local-c4.sh --stop
```
