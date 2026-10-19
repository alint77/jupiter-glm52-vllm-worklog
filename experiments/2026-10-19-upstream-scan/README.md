# 2026-10-19 upstream vLLM scan for tp-sliced-moe

Fork base: upstream main d08eebad16 (2026-07-16). Scanned merged, open, and
closed PRs plus issues up to 2026-10-19 (`git fetch origin main` only, read-only
tree). Target: GLM-5.3 W4A16, DSA/MLA, TP4+DCP4, MTP3 or DFlash2, V2 runner,
FULL_AND_PIECEWISE, fp8_ds_mla, FlashMLA sparse, GH200 sm_90a.

Excluded as already known: #61867, the DCP LSE merge, FlashMLA schedule work,
the prefix-cache miss fix, the cp_gather dtype fix, the DCP indexer Triton
recompile fix, and the MLA q/kv norm+RoPE+KV-write fusion.

Impact numbers marked **guess** are not measured. The rest come from the PR's
own numbers or from our worklog (2026-10-18-c4-swarm-timeline step breakdown).

Spec-method caveat: launchers exist for both MTP3 (`claude-glm53-sliced-mtp3*.sh`,
next_n=4) and DFlash2 (`*-df2-dcp4.sh`, K=7, next_n=8). Several items below
apply to MTP only. Each item says so where it matters.

GLM-5.3-W4A16 `max_position_embeddings` = 1048576. That rules out #58080 (MTP
draft ignores YaRN hf-overrides past native length) for 400K serving.

## 1. Ranked candidates

| # | PR / issue | State, date | What | Why it applies | Est. impact | In tree? | Effort / conflict |
|---|---|---|---|---|---|---|---|
| 1 | [#52164](https://github.com/vllm-project/vllm/pull/52164) | merged 2026-08-14 | DSA indexer native decode for next_n=4 on SM90 (2-CTA KV multicast cluster) instead of flattening to next_n=1 | Our `indexer.py` sets `use_flattening = not SM100 and next_n not in (1, 2)`, so MTP3 (next_n=4) flattens. The bundled `deep_gemm/_C.so` already contains `next_n == 1 or next_n == 2 or next_n == 4` and the `_schedule_meta_size == num_sms / num_kv_multicast + 1` assert, so no DeepGEMM bump is needed. | PR: kernel 1.15-1.36x at 16-32K ctx, bs 4-32. Ours: indexer is about 1.31 ms/step at M=16, so **guess** 0.15-0.4 ms/step, likely more at 100-400K. MTP3 only; DFlash2 next_n=8 still flattens. | No | Low-med. Port `_supports_native_decode` and size the schedule metadata as num_sms/2+1. Check the DCP-local indexer path. The gate keys on the per-step max decode len. |
| 2 | [#59367](https://github.com/vllm-project/vllm/pull/59367) (issue [#59317](https://github.com/vllm-project/vllm/issues/59317)) | open | Bound the DCP global top-k merge scratch (pack + all_gather) to 1024-row chunks via the workspace manager | Reported on our exact shape (GLM-5.3 TP4/DCP4 fp8_ds_mla MTP3, H200). Our `_merge_dcp_topk_global` in `sparse_attn_indexer.py:73-121` all-gathers the full packed candidates. | PR: at T=4096, prefill transient 604 MB -> 144 MiB, about 0.43 GiB less peak. At ~296K KV tok/GiB/rank that is **about 130K KV tokens or about 85 hot experts**, if the profiler run's peak is set by this. Zero decode cost. | No | Low. Self-contained in the indexer. Our CuteDSL pack kernel differs, so this is a hand port. |
| 3 | [#59368](https://github.com/vllm-project/vllm/pull/59368) (issue [#59318](https://github.com/vllm-project/vllm/issues/59318)) | open | MRV2 CUDA-graph memory estimate overcharges (2.16 GiB estimated vs 1.19 GiB actual on GLM-5.3 TP4/DCP4) | Same shape. The fork already skips FULL CG memory profiling under DCP (67e6d48ffd), so check first whether our startup still uses this estimate for PIECEWISE. | Up to about 0.97 GiB/rank of KV (**about 290K tokens**) if the estimate is active, otherwise 0. Confirm against the startup observed-free margin (memory note: about 0.47 GiB). | No | Low. Verify with one startup log before porting. |
| 4 | [#46849](https://github.com/vllm-project/vllm/pull/46849) + [#57443](https://github.com/vllm-project/vllm/pull/57443) (+ open [#61873](https://github.com/vllm-project/vllm/pull/61873)) | merged 2026-08-11 / 2026-10-03 | MRV2 fused multi-step draft decode: build attention metadata once, update it in place per step (`supports_draft_decode_metadata_update`), one CUDA graph for all draft steps | MTP3 runs 3 draft steps with per-step metadata build and launch. PR reports +13.3% at c=1 on GLM-5.3 Hopper. | **Guess** 0.1-0.4 ms of the about 2.10 ms outside-verify time per step. MTP only; DFlash2 is single-pass. | No (`speculator.py` has local argmax but no fused multi-step) | Med-high. Upstream opts in only for FA sparse with dcp==1, so we need FlashMLA-sparse, indexer, and DCP metadata in-place updaters. |
| 5 | [#62174](https://github.com/vllm-project/vllm/pull/62174) (needs FlashInfer #6630/#6631) | open draft | Fuse MoE finalize (top-k weighted sum) into the next allreduce+RMSNorm via `trtllm_moe_finalize_allreduce_fusion` | Our finalize_kernel is about 1.58 ms/step at M=16 and is followed by the TP AR+norm. | **Guess** 0.3-1.0 ms/step at M=16. | No | High. Needs a FlashInfer bump and integration with the sliced kernel's output layout. The fp32 accumulate order changes rounding, so it needs explicit sign-off under same-ops. |
| 6 | [#61253](https://github.com/vllm-project/vllm/pull/61253) | merged 2026-10-12 | GLM tool-call argument streaming goes from O(n^2) to O(chunk) (`arg_value_extender`) | Claude Code Write/Edit tool calls stream 10-100 KB args through `glm47_moe.py`. | PR: 6.4 ms/chunk at 100 KB becomes 7 us. That is API-server CPU, which can stall detokenize and streaming on long writes. | No | Low. Parser only. |
| 7 | Anthropic `/v1/messages` bundle: [#58613](https://github.com/vllm-project/vllm/pull/58613), [#57693](https://github.com/vllm-project/vllm/pull/57693), [#47598](https://github.com/vllm-project/vllm/pull/47598), [#52021](https://github.com/vllm-project/vllm/pull/52021), [#59180](https://github.com/vllm-project/vllm/pull/59180), [#61947](https://github.com/vllm-project/vllm/pull/61947); open [#59617](https://github.com/vllm-project/vllm/pull/59617), [#61356](https://github.com/vllm-project/vllm/pull/61356), [#61920](https://github.com/vllm-project/vllm/pull/61920), [#62293](https://github.com/vllm-project/vllm/pull/62293) | merged Aug-Oct; 4 open | Merged: `thinking: disabled` (Claude Code auto-mode classifier), tool_addition/tool_removal (today a 400 with tool search), named tool_choice to tool_use, disable_parallel_tool_use, count_tokens with thinking/output_config, streaming error events. Open: prompt-too-long error shape (triggers CC compaction), ping keepalive during long prefill, fix for the cherry-picked #58754 probe with tools=None. | Our traffic is Claude Code. Local `anthropic/serving.py` has none of these; billing-header strip and stop_sequence are already present. | Correctness and robustness, no step-time effect. #59617 and #61356 matter most at 400K prefill. | No | Low-med. Many small diffs in one file, which conflict with our local edits to serving.py. |
| 8 | GLM parser: [#56994](https://github.com/vllm-project/vllm/pull/56994) (merged), open [#61312](https://github.com/vllm-project/vllm/pull/61312), [#60842](https://github.com/vllm-project/vllm/pull/60842) (issue [#60703](https://github.com/vllm-project/vllm/issues/60703)), [#49249](https://github.com/vllm-project/vllm/pull/49249) (issue [#49248](https://github.com/vllm-project/vllm/issues/49248)) | mixed | GLM-5.3 always-reasoning detection; a stray `<tool_call>` swallowing the rest of the stream; missing `<arg_value>` handling | Agentic GLM-5.3 traffic through the glm47 parser | Correctness (lost turns), no perf effect | No | Low |
| 9 | Humming W4A16: [#56997](https://github.com/vllm-project/vllm/pull/56997), [#48918](https://github.com/vllm-project/vllm/pull/48918), [#57421](https://github.com/vllm-project/vllm/pull/57421) | merged | Prefer Humming (WGMMA) over Marlin on SM90 for WNA16; CT WNA16 MoE via Humming; shared workspaces | Same weights and same math. Candidates: the untiered MTP drafter MoE (about 1.27 GiB of Marlin) and the prefill MoE. Local humming is 0.1.10, upstream pins 0.1.17. | **Guess**, unmeasured. Needs a bench against Marlin at M=4-16 and at prefill M. | Partial (old package) | Med. Bump the package, then run a kernel bench in `benchmarks/kernels`. |
| 10 | [#56902](https://github.com/vllm-project/vllm/pull/56902) | merged 2026-09-17 | Stale FlashMLA workspace views keep the old storage alive after buffer growth | We grow `q_concat_buffer`/workspaces. A leaked old view pins HBM. | **Guess** tens to hundreds of MB/rank. Check an allocator snapshot for duplicate workspace storage. | No | Low |
| 11 | [#54646](https://github.com/vllm-project/vllm/pull/54646) | merged | Freeze gc during V2 CUDA-graph capture | Startup time across the many arm restarts | PR: large capture-time cut. Startup only. | No | Low |
| 12 | [#58126](https://github.com/vllm-project/vllm/pull/58126) | open | Fuse DCP empty-shard masking (8 kernels into 1) | Applies if our DCP path launches `mask_dcp_empty_shards`-style elementwise ops per layer | **Guess** a few us/layer x 78 layers, so up to about 0.1-0.3 ms/step if present | Check | Low |
| 13 | [#59737](https://github.com/vllm-project/vllm/pull/59737) | open | PDL on the MoE top-k gating kernel | Decode is launch-gap bound in parts | PR about 0.5 us/layer, so about 0.04 ms/step | No | Low |
| 14 | [#61174](https://github.com/vllm-project/vllm/pull/61174) | open | Shared input block-table rows left by a larger batch get read by a larger MTP draft graph. Fix tracks dirty rows and clears them in the gather kernel. | V2 runner with MTP and padded draft graphs. Stale rows can point at freed blocks. | Correctness (latent bad KV reads or writes) | No | Low (block_table.py) |
| 15 | [#60736](https://github.com/vllm-project/vllm/pull/60736) | merged | MRV2 int64 index/slot mappings | Overflow needs int32 slot * stride above 2^31. At our sizes the threshold is about 5368 slots at 400K, so low risk. | Robustness | No | Low |
| 16 | [#59204](https://github.com/vllm-project/vllm/pull/59204) | open | DeepSelect top-k on sm90 | Alternative to our cooperative/persistent top-k (`sparse_attn_indexer.py:575-615`). Same output set. | **Guess** small, needs a bench vs persistent at 400K | No | Med |
| 17 | [#53407](https://github.com/vllm-project/vllm/pull/53407), [#50488](https://github.com/vllm-project/vllm/pull/50488), [#58400](https://github.com/vllm-project/vllm/pull/58400) | merged | MRV2 cudagraph dispatch and padding fixes | We set capture sizes by hand ([8] or [1,8]), so we likely never hit the bad dispatch | Low | No | Low |
| 18 | HiSparse ([#53781](https://github.com/vllm-project/vllm/pull/53781), [#56629](https://github.com/vllm-project/vllm/pull/56629), et al.) | merged | Host-resident sparse-MLA KV, GPU hot LRU, in-kernel resolver | Idea only for KV pool overflow onto Grace over C2C | Idea | No | Very high (+14K lines). Shared pool needs DCP==1 and rejects FULL cudagraph. Adds per-step resolve work, which breaks the zero-step-overhead rule. |
| 19 | [#62000](https://github.com/vllm-project/vllm/pull/62000) | open | Candidate-only FP8 indexer scoring on SM90 (DSv4.1) | Idea only. It changes which tokens get scored, so it is likely an algorithm change. | Idea | No | Not allowed unless output-identical |
| 20 | [#54394](https://github.com/vllm-project/vllm/pull/54394), [#58536](https://github.com/vllm-project/vllm/pull/58536) | open | Indexer row sharding across TP | Moot under DCP4, since DCP already splits the indexer KV | 0 | No | n/a |

Already covered in our tree (no action): #59319 (prefill workspace 0 under DCP,
`flashmla_sparse.py:715-730`), #58985, #59464, #58169, #49793 (local argmax),
the indexer prefill buffer sizing (fork f3c882485e), and #61285 (DCP a2a; we use
one-shot).

## 2. Bug fixes we likely need

- **#59317 / #59367**: DCP top-k merge transient OOM risk on our shape at long
  prefill chunks (item 2).
- **#59318 / #59368**: CG memory overestimate (item 3). Verify that it is active first.
- **#56902**: stale FlashMLA workspace views retain HBM (item 10).
- **#61174**: stale shared block-table rows read by a larger MTP draft graph (item 14).
- **#60450 / #60493**: `expanded_block_table` allocated after KV sizing, so it is
  not accounted. Small for us (about 25 MB) but it erodes the 0.47 GiB margin.
- Spec decode:
  - [#50183](https://github.com/vllm-project/vllm/pull/50183): NaN logits lead to out-of-bounds argmax.
  - [#48630](https://github.com/vllm-project/vllm/pull/48630): rejection sampler chunking.
  - [#47616](https://github.com/vllm-project/vllm/pull/47616): a stop string under spec decode trims token_ids wrongly.
  - [#59260](https://github.com/vllm-project/vllm/pull/59260): warm up the rejection kernels to avoid a first-request stall.
  - [#61451](https://github.com/vllm-project/vllm/pull/61451): skip bonus reductions; small perf gain.
  - [#61161](https://github.com/vllm-project/vllm/pull/61161).
- Not ours, noted for completeness:
  - #56734: dummy draft steps write KV through stale rows; DP only.
  - #58784: P/D only.
  - #61496 / #55212: GQA+DCP FULL graph; MLA unaffected.
  - #58080: MTP YaRN; our max_position_embeddings is 1M.
  - [#59305](https://github.com/vllm-project/vllm/pull/59305): FlashMLA o_accum on SM90. Our pin a8f794d predates both the regression and the fix, and decode T is small.

## 3. Notable rejected / closed ideas

- **QREP #45964 / #50382** (merged upstream, bad trade for us): it replicates q_b
  to drop an all-gather. That costs about +3.9-4.6 GiB/rank and 4x q_b weight
  streaming per step, which is more than the all-gather it removes at bs<=16.
- **#52696** FP16 indexer logits: closed with conflicts, and it changes the math.
- **#39943** multi-stream overlap of the indexer wk: went stale. We already run
  the indexer on a side stream.
- **#39941** CUTLASS FA3 sparse MLA for Hopper: stale. It has no fp8_ds_mla or DCP support.
- FA3-sparse and FlashInfer SM90 sparse-MLA backends: neither supports fp8_ds_mla+DCP.
- **#57206** DeepSelect for prefill: SM100 only, closed.
- **#60029** 64-head padding: SM100 only.
- **#62102**: plain fp8 KV only. **#57701**: Flash/kpool only.
- **#58578** reduced draft vocab: forbidden (same ops).
- **#41467** MTP EOS truncation detector: bot-authored, unmerged, dubious root cause.
- **#49183** disable DeepGEMM on GH200: open and unverified. It is a MiniMax FP8
  GEMM issue that does not touch our indexer kernel.
- Superseded or duplicate:
  - #51936 was split into smaller PRs.
  - #48528 was superseded by #50904.
  - #48897 was superseded by #50484.
  - #49357 was superseded by #53755.
  - #47617 was fixed by #44993.
  - #43114 went stale.

## 4. Searched, nothing relevant

- **GH200 / Grace / UVA / C2C**: only generic UVA offloader fixes. #62060 (UVA
  expert staging) resembles our tiered MoE and has nothing new.
- **Symmetric memory and custom allreduce**: #61989, #56891, and #52998 tune
  Blackwell/MNNVL/NVSwitch only.
- **PDL**: nothing beyond #59737.
- **Marlin W4A16**: open PRs are W4A8/FP8 fixes only.
- **Scheduler**:
  - #54627 (prefill_schedule_interval) overlaps with our VLLM_SCHED_INTERLEAVE_SRPT.
  - #57951 and #58459 add a long-prefill threshold.
  - #56758 adds max-num-active-seqs.
  - Open SRPF ideas: #61741, #54145, #58420.
  - None beats what we have.
- **FlashMLA bump #61371**: SM90 V3.2 sparse kernel unchanged.
- **DeepGEMM bump** (upstream fork a7a92773 vs our a6b593d2): the next_n=4 SM90
  kernel is already in our bundled .so.
- **TokenSpeed MLA**: Blackwell only.
- **KV offload connectors and prefix-cache tiers**: out of scope.
- **FP8/FP4 weight paths and KV formats below fp8_ds_mla**: excluded by constraint.

## Suggested next steps (cheapest first)

1. Read one startup log to check whether the #59368 estimate and the #56902
   retained views apply, and take an allocator snapshot.
2. Port #52164 for MTP3 arms. Bench the indexer at next_n=4 against flattening at 100K and 400K.
3. Port #59367 and measure the startup observed-free delta.
4. Port the parser and Anthropic fixes (#61253, #56994, #60842, #58613, #57693, #59617, #61356) as one serving-correctness batch.
5. Bench Humming 0.1.17 against Marlin for the drafter MoE.
6. Scope #46849/#57443 (MTP) and #62174 (needs a FlashInfer bump and same-ops sign-off) as larger projects.
