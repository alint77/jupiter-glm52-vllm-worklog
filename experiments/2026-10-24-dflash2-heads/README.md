# New DFlash2 heads at k=3 vs MTP3 and the incoai head, on the c=4 Claude Code replay (2026-10-24)

Two more GLM-5.3 DFlash2 drafters are on HF. The question is whether either one beats MTP3 (prod) or the incoai head on our traffic. On our traffic, the incoai head trailed MTP3 at k=3: ~4% lower acceptance on code, more away from code. See the write-up, `glm/README.md` sections 10-11, and `../2026-10-09-ood-accept`.

## The heads

All three are bf16. They are served as the drafter at `fp8_per_channel` weights and fp8 KV (serve.sh's DFlash2 defaults, as prod ran DFlash2).

| head | local dir (`/e/fscratch/profound/naeimitabiei1/models/`) | HF revision |
|---|---|---|
| incoai/GLM-5.3-DFlash2 (used so far) | `GLM-5.3-DFlash2` | downloaded 2026-08-28 |
| SubconsciousDev/glm-5.3-dflash2 | `GLM-5.3-DFlash2-SubconsciousDev` | 7288fc2b4c63 |
| AlayaNeW/GLM-5.3-DFlash2 | `GLM-5.3-DFlash2-AlayaNeW` | 5a6aca9a10c2 |

**SubconsciousDev** is a fine-tune of the incoai head:
- Its `config.json` is identical to incoai's, and its README is incoai's, copied.
- 21 of 96 tensors are byte-identical to incoai's: the candidate selector, `fc`, the final norm and some layer norms.
- The other 75 tensors (attention, MLP, the conv kernels) differ.
- It shares no tensor with the `GLM-5.3-DFlash2-v2` checkpoint we also hold.

**AlayaNeW** is a different model. Compared with incoai's 6 sliding-attention layers, 8 KV heads and target layers [5, 19, 33, 47, 61, 75]:
- 5 layers: 4 sliding-window (2048) and 1 full attention;
- 16 KV heads;
- target layers [1, 20, 38, 56, 75], so `fc` is 30720 → 6144;
- block size 12, mask token 154821, rope theta 8e6;
- trained with a "dpace" objective (`deepspec_dflash2` block).

Its README reports acceptance at 8 draft tokens on SPEED-Bench, measured on H200 with an FP8 target.

## Setup

- Frozen worktree `worktrees/df2h` = vllm 23d2796ed5. This is prod: the main tree's uncommitted changes are memory probes and a no-op `#define`.
- The c=4 prod config, as `claude-glm53-sliced-mtp3-c4.sh`:
  - TP-sliced MoE, touch-order hot set, MLA qkv glue on;
  - 4 seqs on a 0.8M pool, SRPT, reserve 3.6 GB;
  - capture sizes 1-8, 12, 16 plus the prefill sizes.
- Every arm uses the same reserve, so the verify step is the same apart from the drafter.
- Load: the 600 s c=4 closed-loop replay of the long Claude Code session (`../2026-10-16-mtp-depth-c4/timeline_save.py`, no profiler).
- Analysis: `../2026-10-16-mtp-depth-c4/ab_depth.py`, giving per-position acceptance from /metrics and the per-turn decode rate, plus a comparison on the turns both arms reached.

Usage:

    run.sh <hold> <run> <arm>...    arms: mtp3 inco sub alaya   (DUR=<s> shortens)
    arm.sh                          ../2026-10-16-mtp-depth-c4/arm.sh with SPEC overridable

serve.sh gained `DRAFTER=<dir>`, which overrides the DFlash2 head; the default is unchanged.

Run h1 places the arms in a ring over four holds, so every arm runs on two nodes and has a same-node neighbour:

| hold | arms |
|---|---|
| 2493858 | mtp3, inco |
| 2493859 | inco, sub |
| 2493860 | sub, alaya |
| 2493861 | alaya, mtp3 |

Outputs go to `/e/fscratch/profound/naeimitabiei1/sliced-prof/h1-<arm>-<hold>/`.

## Results

(pending)
