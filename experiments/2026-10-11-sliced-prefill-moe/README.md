# Sliced prefill MoE kernel: toward the floor at sliced shapes

Shipped as vLLM `c8d61506be` ("Tiered prefill MoE: faster sliced prefill (4K
chunk 2594 -> 1974 us)"). Kernel: `vllm/model_executor/layers/fused_moe/
tiered_prefill/tiered_prefill.cu`; context and kernel structure in
`REVIEW_BRIEF.md` (the brief given to Astra).

## Result

Isolated, one GH200, real routing, CUDA graph (`bench_sliced.py`). Every
version is bitwise identical to the previous main tree on 14 calls
(T 1..4096, `bitwise.py`).

| chunk | before | v9 (shipped) |
|---|---:|---:|
| 512 | ~652 us | 604 |
| 1K | 884 | 807 |
| 2K | 1416 | 1181 |
| 4K | 2594 | 1974 (-24%) |

4K timeline (us): route 19, gather 142, w13 ~1060, act 34, w2 ~630, combine
134. w13 runs at ~80% and w2 at ~71% of the practical tensor peak on their
padded work; the narrow (memory-bound) classes leave ~160 us of tail after the
wide ones.

## What did it

- v1-v6: w2 keeps a unit's X resident in smem and streams W past it (`TP_XRES`,
  NT >= 80, 8 blocks/CTA); unit-major block order; faster gather/act.
- v7: finer wide classes 80/112 (`TP_FINE_CLASSES`, T >= 1536 only): less
  padding; -38 us at 4K, -19 at 2K.
- v8: route no longer writes `route_of_pos` (unread; 32K scattered stores from
  one SM): route 41 -> 19 us at 4K.
- v9: `wait<1>` on the w13 wgmma pipeline for NT 112/128 (~-20 us at 4K);
  gather undefined read (Astra) fixed without a stack frame.

## Pitfalls found

- ptxas C7515: wgmma serializes when non-wgmma code defines accumulator
  registers mid-pipeline. With `wait<1>` that hits NT 80/96, and any runtime
  branch in the mainloop (the gather-mode flag): gather mode is now a template
  parameter.
- Dynamic register-array indexing (`pos[threadIdx.x]`) gives a stack frame;
  use direct loads.
- Cross-run sweeps are +-25 us noisy; `bench_var.py` loads define variants side
  by side in one process and interleaves replays. Run it in both orders.

## Tried and dropped

- Warp-aggregated atomics in route (`__match_any_sync`): 2x slower.
- Gather fused into w13 (pfk worktree v10, `TP_GATHER_FUSE`): bitwise equal,
  +150 us at 4K and slower at every size; wide kernels lose more loading X
  row by row than the gather saves. Not ported.
- w2 xres blocks/CTA sweep: flat, kept 8.
- Green contexts for the narrow tail: ruled out earlier (2026-07-27), forced
  co-residency cost more than it overlapped.

## Open levers

Narrow-class tail (~160 us at 4K), combine (134 us), gather (142 us), w2
efficiency (71%).

## Files

`run.sh` (node runner, GPU-local NUMA, per-tree extension cache), `bench_sliced.py`,
`bench_var.py`, `bitwise.py`, `ncu.sh`/`ncu_src.sh`/`ncu_table.py`/`src_top.py`
(ncu by kernel and by source line), `bench_sched.py`/`sched.sh` (unit order),
`probe_narrow.py`, `logs/`.
