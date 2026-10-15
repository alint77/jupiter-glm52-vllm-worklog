# MTP5 vs MTP3 at c=4 on the agentic timeline (2026-10-16)

Question from the mlx.fast Bonsai track (`../2026-10-16-cudafast-sidequest`):
MTP3's last draft position is accepted in 54.7% of rounds (live counters), so
the draft is censored. Would MTP5 collect more than its extra verify rows
cost? On paper about break-even: +26% tokens/round if acceptance holds at 0.8
per position, against about +23% step time at ~0.75 ms per row (c=4 M=16 -> 24).

Each arm runs the c4 launcher config (`arm.sh` = `../2026-10-11-nsys-fresh/arm.sh`
with SPEC_K and CAPTURE_SIZES overridable) on worktree pf5 (144f2f4aee), FCFS,
on its own fresh hold. The load is the 600 s c=4 closed-loop replay
(`timeline_save.py`: the c4-timeline driver plus saved outputs and /metrics
snapshots). Both arms capture 1-8, 12, 16, 18, 24, ... .

    run_queue.sh <hold>...          # queue/k3 k5 k5-r2 k3-r2; holds never cancelled
    ab_depth.py <arm dir>...        # tok/s, per-turn decode rate, acceptance per position
    lookup_glm.py <arm dir>         # prompt-lookup coverage on GLM's own outputs

Holds 2344299 (k3) and 2344298 (k5); results under
/e/fscratch/profound/naeimitabiei1/sliced-prof/mtpd-<arm>-<job>/.
