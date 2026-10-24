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

The worktree also carries the main tree's ignored build artifacts (`git status --ignored vllm`: the `.so` extensions, `vllm_flash_attn/{cute,layers,ops}`, `third_party/*`), copied in. MTP arms never import FlashAttention, but the DFlash2 drafter does. Without them the first smoke test (hold 2492777) failed with `vllm.vllm_flash_attn requires the CUDA flash attention extensions`.

## Results (run h1 on 4 holds + run h2 on hold 2492597, 2026-10-24)

These are 600 s c=4 replays. Tokens/step is 1 + accepted/drafts from /metrics. The per-turn decode rate is output tokens from the first token to the end of the turn, median over turns. ms/step is tokens/step divided by that rate. It is derived, so it includes the steps where other agents' prefills are interleaved.

| arm | node | tokens/step | accept per position | per-turn decode (median) | ~ms/step |
|---|---|--:|---|--:|--:|
| MTP3 | 065-15 | 2.764 | .761 .572 .431 | 105.8 | 26.1 |
| MTP3 | 106-18 | 2.780 | .767 .576 .438 | 106.1 | 26.2 |
| MTP3 | 044-46 | 2.819 | .774 .592 .454 | 107.6 | 26.2 |
| incoai | 065-15 | 2.607 | .719 .517 .370 | 101.5 | 25.7 |
| incoai | 065-16 | 2.628 | .726 .523 .379 | 99.6 | 26.4 |
| SubconsciousDev | 065-16 | 2.622 | .723 .520 .379 | 98.8 | 26.5 |
| SubconsciousDev | 106-17 | 2.632 | .726 .527 .379 | 97.3 | 27.0 |
| SubconsciousDev | 044-46 | 2.630 | .725 .525 .380 | 98.6 | 26.7 |
| AlayaNeW | 106-17 | 2.565 | .701 .500 .364 | 90.8 | 28.3 |
| AlayaNeW | 106-18 | 2.554 | .699 .496 .358 | 92.2 | 27.7 |
| AlayaNeW | 044-46 | 2.563 | .704 .499 .360 | 90.4 | 28.4 |

Same-node per-turn decode ratio on the turns both arms reached (median / geomean):
- incoai / MTP3: 0.946 / 0.994
- SubconsciousDev / incoai: 0.985 / 0.912
- AlayaNeW / SubconsciousDev: 0.925 / 0.904
- AlayaNeW / MTP3: 0.856 / 0.890
- h2 (044-46): SubconsciousDev / MTP3 0.952 / 0.908; AlayaNeW / MTP3 0.879 / 0.869

Findings:
- **MTP3 stays ahead.** Its first draft position accepts 0.76-0.77, against 0.70-0.73 for every DFlash2 head. It gets ~5% more tokens per step at the same step time, so per-turn decode is ~5% faster than incoai, as in the write-up.
- **SubconsciousDev is the incoai head.** Tokens/step and per-position acceptance match incoai's to the third decimal on the same node, so the fine-tune changed nothing measurable on this traffic. Its decode rate is level to ~2% lower.
- **AlayaNeW is worst on both counts:**
  - it accepts less (2.56 tokens/step);
  - its step is ~1.5-2 ms longer. Its drafter has a full-attention layer over the whole context, and twice the KV heads, where the others use only 2048-token sliding windows. These contexts run 50-150K.
  - Net: -10 to -14% per-turn decode against MTP3.
- **Memory:** DFlash2 arms start with 4.9 GiB free at the shared 3.6 GB reserve (MTP3 ~2.5). A DFlash2 deployment could turn that into ~100 more hot experts per GPU. That is worth well under the 5% acceptance gap: the write-up's c=4 DFlash2 at its own reserve 1.7 was level with MTP3 at best.
