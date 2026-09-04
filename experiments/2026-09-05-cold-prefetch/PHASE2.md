# Phase 2: staging plumbing, in situ

**Pass.** Job 1667639 on `jpbo-044-17`, the production MTP3/c=1/DCP1/400K
config with `VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024` and
`..._VERIFY=1`. Every layer's cold tier is copied into the HBM slot and the
staged bytes are compared against their Grace source; nothing reads the slot,
so the cold tier still executes from Grace and output must be unchanged.

## Result

```
cold prefetch: chunk 14 staged 48.5 GiB across 75 layers, 1036 verified, 0 mismatched
```

| | |
| --- | ---: |
| chunks staged | 14 |
| layers per chunk | 75 |
| staged per chunk | 48.5 GiB (52.1 GB) |
| byte compares | **1036** |
| mismatches | **0** |

Semantics unchanged, as they must be when nothing reads the slot: the probe
returns `" Paris. Currency -- the Euro. France"` and the ~96K prompt returns
real completion text. Server started, served, and shut down cleanly.

Free HBM after load is 10.11 GiB (87522 of 97871 MiB used), consistent with the
10.33 GiB the reserve check reported. The 830 MiB slot is allocated lazily on
the first `apply_tiered_moe`, so it is **not** covered by the load-time reserve
check -- phase 4's `cold_staging` accounting is what fixes that, and it is not
optional before anyone runs this at production reserve.

## What this phase caught

A launch failure, which is what it was for:

```
ValueError: Expert backing has 424673440 bytes, expected 389284000
```

The coordinator built its slot views from a hardcoded group size of 128. This
checkpoint is compressed-tensors at **group 32**, where the two scale
components are four times larger. The layout has the right shape and the wrong
length, so nothing but `build_expert_component_views`' own backing-size assert
stood between it and silently misaligned weights.

Fixed by mirroring each tier's own `component_specs`, which
`ExpertTierStorage` already carries -- the coordinator no longer knows what a
group size is. The tests are now parametrised on group size and were confirmed
to fail against the hardcoded version at 32 while passing at 128.

## What the unit tests do not cover

Cross-stream ordering. The test tiers are ~93 MiB and the copy lands before the
compare whether or not `await_staged` is called -- removing it leaves every
test passing, which was checked rather than assumed. This run is the ordering
evidence: 830 MiB layers, 1036 compares, zero mismatches.

## Next

Phase 3 switches the cold tier to read the staged views, gated on bitwise
identity against this arm.
