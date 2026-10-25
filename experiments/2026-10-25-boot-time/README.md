# Why the prod server takes ~20 min to boot, and what would speed it up (2026-10-25)

The c=4 TP-sliced MTP3 Claude Code server (`claude-glm53-sliced-mtp3-c4.sh`) was booted today as job 2502713. It took **19 min 32 s** from job start (10:45:41) to ready (11:05:13), after 6.7 min in the queue. Its log is `../2026-09-27-glm53-mtp7-profile/server-claude-2502713.{out,err}`.

This experiment breaks down where that time goes, and measures the weight load on a fresh hold (2503094, jpbo-096-37).

## Cold boot (prod job 2502713)

| phase | time | cause |
|---|--:|---|
| job start → vLLM banner | 2:00 | Before `serve.sh`, Slurm batch startup is probably ~1 min (see the probe below). Then environment setup and the CLI imports. Earlier boots spent 1.2-2.9 min here. |
| config, EngineCore and worker spawn, NCCL | ~1:15 | Python imports in each process (see below) |
| `one_shot` extension build | 0:48 | one-off rebuild |
| weight loading | 6:24 | mmap page faults (see below) |
| torch.compile | 4:17 | **cache miss** (a hit takes 36-45 s) |
| profile run | 1:38 | includes the `mla_glue` (44 s) and `tiered_prefill` (50 s) extension builds |
| drafter compile + warmup | 0:15 | |
| `decode_gemm` build, KV setup | ~0:53 | |
| `anchor_hot` build | 0:42 | |
| warmups + graph capture (32 s) | ~0:45 | |
| API start + health | ~0:20 | |

Two of these are avoidable:

- **The compile-cache miss recurs.** The launcher runs prod from the live main tree. The compile cache is keyed on a hash of the source code, so the first boot after nearly any commit misses (350-360 s against 38-45 s on a hit). The prod cache root holds 7 hash directories, one per code version.
- **The extension rebuilds (~3.8 min) were one-off.** All 5 torch cpp_extension JIT builds (`one_shot`, `mla_glue`, `tiered_prefill`, `decode_gemm`, `anchor_hot`) rebuilt, one after another. ninja's dependency lists include system headers, and the cluster OS update changed the mtimes of `/usr/include/linux/*` to 2026-10-19 12:46, on login and compute nodes alike. Editing an extension's own `.cu` file also triggers a rebuild.

## Warm boot, measured (`node_probe.sh` on hold 2503094)

Setup: the prod env from the main tree, on a copy of the prod cache root, so compile and extension caches hit. `PYTHONPROFILEIMPORTTIME` is on, vmstat runs every 2 s, and py-spy samples the TP0 worker at 10 Hz during the weight load.

Result: **10 min 56 s** from `serve.sh` to the API up (13:57:04 → 14:08:00).

| phase | time |
|---|--:|
| `serve.sh` → banner (environment setup 1.8 s, CLI imports) | 0:55 |
| config → EngineCore → workers, NCCL, planner | 1:15 |
| **weight loading** (target 352 s + MTP layer 19 s, plus setup) | **6:21** |
| torch.compile (cache hit) | 0:36 |
| profile run | 0:33 |
| KV setup + warmups | 0:17 |
| graph capture | 0:37 |
| API start | 0:14 |

Prod's pre-banner 2:00 is ~1 min longer than the probe's 0:55. The difference falls before `serve.sh` runs, presumably Slurm batch-step startup. The logs do not cover it.

### Weight loading is bound by mmap page faults, not GPFS or the GPU

- **py-spy, TP0 main thread** (3,724 samples over the load):
  - 69% is `transposed()` in `convert_glm_w4a16_expert`. This is the CPU `.t().contiguous()` of each tp_sliced expert slice; it is what first touches the mmap'd checkpoint.
  - 15% is safetensors `__getitem__`.
  - Marlin repack and permute on the GPU are about 1%.
- **vmstat:** the CPUs are 99% idle throughout. The page cache grows at **1.23 GB/s**, which is the load rate.
  - The 4 ranks walk the same files together, so the node reads the 420 GiB checkpoint once, and each rank faults its slices in, one at a time.
- **GPFS `read()` bandwidth** (dd, `gpfs.txt`):

  | readers | GiB/s |
  |--:|--:|
  | 1 | 6.8 |
  | 4 | 24.2 |
  | 4, same files | 24.3 |
  | 16 | 43.7 |
  | 32 | 41.1 |

  So storage has about 20× headroom over the load rate.
- **One shard through the loader's slicing** (`shard_probe.py`, `logs/shard-probe-2503094.log`):

  | read path | 1 process | 4 processes, same shard (as the 4 ranks) |
  |---|--:|--:|
  | mmap | 0.72-0.76 s | 1.12-1.33 s |
  | eager: one `read()` of the file | 0.54-0.60 s | 0.58-0.62 s |

  - 282 shards × 1.2 s ≈ the 340 s observed for the target load.
  - A `read()` of a shard does not warm the page cache for a later mmap much (0.52 s, against 0.21 s for a fully resident shard), so a read-ahead prefetch thread does not help the mmap path.

### Eager loading (`--safetensors-load-strategy eager`)

- **As is, it fails.** The eager branch of `safetensors_weights_iterator` yields tensors in the order of the deserialized dict. The tiered stager needs each expert's components in a row, and it raises `Checkpoint expert components are interleaved or incomplete`.
- **The fix:** iterate `sorted(state_dict.items())`. That is the order the mmap path's `safe_open.keys()` already yields (checked: `keys() == sorted(keys())`). It is the same tensors in the same order, so the loaded weights are byte-identical.
- The fix was tested in the frozen `worktrees/df2h` only (uncommitted), boot `out-2503094-eager2`:

  | | mmap (prod) | eager (sorted) |
  |---|--:|--:|
  | target weights | 352.2 s | **158.3 s** |
  | of which expert conversion | 265.7 s | 51.0 s |
  | MTP layer weights | 19.2 s | 109.2 s |
  | model loading total | 381.0 s | 273.0 s |

- **The eager boot came up healthy:** 2.88 GiB observed free, against 2.91 GiB for the mmap boot; graph capture took 31 s. It still took 12:18 overall, because running from `df2h` changed the source paths. All 5 extensions rebuilt (the profile run took 131 s, against 33 s), and the drafter's AOT artifact failed to load and recompiled. Neither has anything to do with eager loading. It also shows that caches are tied to the path of the tree as well as its code, so a pinned prod worktree must keep one fixed path. The patch is `eager-sorted.patch`; `df2h` is restored to 23d2796ed5.
- **The MTP pass regresses.** Its second load pass re-reads all 282 shards in full under eager, where mmap only touched the MTP layer's own tensors.
- **Eager for the target and mmap for the drafter** would give ~177 s, **-3.4 min**.
- What remains of the 158 s is mostly ~280 serial whole-shard reads at ~3.3 GB/s each. Reading the next shard on a thread while the current one converts would approach max(read, convert) ≈ 110 s. Parallel chunked reads, which GPFS serves at 24-44 GiB/s, would go lower still.

### Imports

- `import vllm.entrypoints.openai.api_server` takes **22.5 s** with a warm cache (`importtime-warm.txt`).
  - 13.8 s of that is under `vllm.config`.
  - **8.9 s is self time in `transformers.utils.import_utils`.**
- The API server, EngineCore and the 4 workers each import vLLM, and the chain is serial (API server → EngineCore → workers), so this is roughly a minute of every boot.

## What would speed it up (same math throughout)

| lever | saves | cost |
|---|--:|---|
| Run prod from a pinned worktree commit, and prewarm its cache on a hold after each change | ~4 min whenever the code changed since the last boot | launcher change |
| Prebuild the 5 JIT extensions ahead of time, in parallel (or as one build) | up to ~3.8 min when stale (header updates, `.cu` edits) | build script |
| Eager load for the target model (sorted fix) and mmap for the drafter | ~3.4 min, every boot | small loader change, needs review |
| Pipeline or parallelize the shard reads | a further ~1-1.5 min | loader change |
| Trim the 8.9 s `transformers.utils.import_utils` import, per process | ~0.5 min | investigate |
| Slurm pre-`serve.sh` time | ~1 min, unknown cause | not ours |

With the first three, a boot is ~7.5 min, against 19.5 min today and 11 min warm.

## Files

- `probe.sh <hold>`: copies the prod cache root (one copy per hold, deleted afterwards) and runs `node_probe.sh`. `TAG`, `LOAD_EXTRA` and `SKIP_RD` select a variant.
- `node_probe.sh`: the boot with import times, vmstat and py-spy, then the GPFS dd probes.
- `shard_probe.py`, `shard_probe.sh`: one shard through the tp_sliced slicing, mmap against eager.
- `out-2503094/` (warm prod boot plus probes), `out-2503094-eager/` (failed eager boot), `out-2503094-eager2/` (eager, sorted fix), `logs/`

## Levers shipped (2026-10-25, afternoon)

- **Read-ahead load** (vllm f722d708e2, local).
  - The tp_sliced target reads up to 2 shards ahead in 64 MiB parallel `preadv`s and yields zero-copy views. The order is the mmap walk's: files naturally sorted, names sorted.
  - `tests/model_executor/model_loader/test_tiered_moe_readahead.py` checks the stream against mmap.
  - On real GLM shards, 1,244 tensors (bf16, f32, i32, i64) are byte-identical.
  - `VLLM_TIERED_MOE_READAHEAD_LOAD=0` restores mmap.
- **The drafter skips the target's experts** (vllm a49557e242).
  - With the target no longer filling the page cache, the tp_sliced MTP drafter's mmap walk faulted the whole checkpoint in again: 109 s on a fresh node (boot A).
  - It now gets the layer-aware expert map the EP drafter uses (none of the target's experts, all of its own), so the target's routed experts are skipped before any read.
  - Two tests in `test_registry.py` fail with and without this change: their fake configs lack `tiered_moe_config.layout`. They predate this work.
- **Parallel prebuild:** `agent_space/scripts/prebuild_extensions.py`, started in the background by `serve.sh`.
  - It builds the 6 JIT extensions at once with no GPU visible. `TORCH_CUDA_ARCH_LIST=9.0+PTX` reproduces `one_shot`'s device-derived flags; build.ninja is identical to the server's own.
  - Timing: 88 s for a full build, 19 s for an up-to-date check, all behind the weight load.
- **Pinned prod tree:** `agent_space/scripts/prod_pin.sh [commit]` sets up `worktrees/prod` with the built artifacts copied in.
  - `claude-server.sbatch` runs from it via `PYTHONPATH` (`CLAUDE_PROD_TREE=` opts out).
  - `prod_prewarm.sh <hold> <c>` boots a launcher's config once to warm its caches.

Boots on hold 2507069 (jpbo-004-01), all from `worktrees/prod`:

| boot | cache | to healthy | target load | MTP load | torch.compile | profile run |
|---|---|--:|--:|--:|--:|--:|
| A | copy of prod c=4 cache, new tree path (miss) | 14:17 | **64.7 s** | 108.8 s | 248 s | 94 s |
| B | same copy, warm | **8:56** | 64.8 s | 18.6 s | 32 s | 120 s |
| C | real prod c=4 root (prewarm) | 11:51 | 64.2 s | 6.6 s | 241 s | 94 s |

- "To healthy" counts from `srun` start and includes ~1.5 min of imports and spawn.
- The target load fell from 352 s to 65 s.
- In B and C the MTP load benefits from the page cache of the earlier boot on the same node.
- A profile run of 94-120 s, against 33 s in the warm mmap boot: the prebuild was still checking or building when the profile run's `load()` calls ran, and they waited on its locks. That needs a look.
