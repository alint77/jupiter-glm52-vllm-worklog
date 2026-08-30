# Does buying HBM residency pay? (2026-08-30)

Open since Phase 32 and never actually tested: a profile's slot count does not
bind residency unless `VLLM_TIERED_MOE_PROFILE_CAP=1`, which nothing sets, so
every earlier attempt compared *rankings* at identical residency. Phase 46's
"hot-slot sweep" was withdrawn for exactly this.

Four arms, one allocation (job 1536855), real-code decode suite, c4/DCP4/MTP3,
W4A16. The three capped profiles are built from the same traces at different
`--hot-slots-per-rank` and their hot sets are verified **nested**
(1800 ⊂ 2100 ⊂ 2400), so the ranking is fixed and only the count moves.

Residency is now logged, so these are the counts the server actually ran, not
counts inferred from a filename:

| arm | logged residency/rank | tok/s | TPOT (ms) |
| --- | ---: | ---: | ---: |
| cap1800 | 1800 hot / 3000 cold | 165.87 | 23.12 |
| cap2100 | 2100 hot / 2700 cold | 176.87 | 21.79 |
| cap2400 | 2400 hot / 2400 cold | 193.73 | 19.73 |
| uncapped | **2466** hot / 2334 cold | **195.14** | 19.44 |

**Buying residency pays: +17.6% from 1800 to 2466, monotonic**, with the
marginal value falling off at the top (2400 → 2466 is +0.7% for 66 experts).
Production sits at the ceiling already: uncapped means "as many as HBM allows",
and the log says that is 2466 at 48.8 GiB available and 20.3 MiB per expert.

Two corrections this produces:

- **Production runs 2466 hot experts per rank, not the ~2730 derived from the
  memory budget on 2026-08-30.** That estimate was 11% high, which is why the
  count is logged now rather than computed.
- The shipped profile lists 2496 and is *demoted* to 2466; the 2400 profile is
  *promoted* to it. Neither number is the residency.

**The lever is HBM, not the profile.** Since residency is already at the
ceiling, more throughput from this direction requires freeing HBM — which also
prices the tier-overlap work negatively, because its 1.6 GB per rank of extra
workspace would cost roughly 79 hot experts.
