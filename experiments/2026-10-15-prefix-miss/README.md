# Append-only prefix-cache misses (~35K cap)

2026-10-15. Started from `2026-10-15-c4-timeline` (the 30-70 s TTFT turns at
c=4 are mostly these misses).

## Symptom (c=1, fresh server, empty pool; `../2026-10-15-c4-timeline/logs/fix-pfx-2334600.log`)

| request (run order) | prompt | hit |
|---|---|---|
| #112 | 240,176 | 0 |
| #113 | 240,430 | 239,872 (99.8%) |
| #114 | 241,223 | 37,376 |
| #16 / #17 | 68,738 / 76,334 | 35,072 / 35,072 |
| #63 / #64 | 155,927 / 156,492 | 36,096 / 36,352 |
| #160 / #161 | 294,939 / 301,753 | 24,320 / 22,784 |

Every earlier turn of the session is a prefix of the later ones (so #16
should hit ~68K from #112's blocks).

## Ruled out

- **Server-side prompt.** `render_diverge.py` on a GPU-less render server
  (`vllm launch render`, same parsers as serve.sh, `render-server.log`) fed
  the server's own Anthropic->OpenAI conversion: #112->#113->#114, #16->#17,
  #63->#64 are 100% token prefixes of each other (`renders/`).
- **Tokenizer.** fastokens == HF token-for-token on these prompts.
- **Pool pressure / concurrency.** c=1, 800K-token pool, KV usage 0 after
  each request.
- **Bare KVCacheManager mechanics** (`cpu_repro.py`): cold A, A+254, A+1047
  with chunked prefill, MTP lookahead decode steps and free, at layouts
  unitary / MLA+indexer / +drafter groups (eagle on/off), DCP4, hash block
  256 and 64: ~99% hits everywhere. Does not reproduce.

## Astra (gpt-6-astra, `astra-1.md`)

Ranks (1) a reused block's mapping damaged during #113's allocation/commit/
cleanup, (2) the real group layout exposing a lookup/adoption inconsistency
(other full-attention groups not trimmed to the reconciled hit; DCP block
size vs replicated drafter groups), (3) hash-identity change despite equal
tokens. Best probe: snapshot #113's reused `(group, hash) -> block_id` and
diff at #114 at the first hole: key changed / mapping gone / mapping valid but
finder stops / block metadata disagrees.

## Probe (worktree pf4 = 877df9b262 + debug, not for commit)

`VLLM_DEBUG_PREFIX_DUMP=<dir>`: at every `get_computed_blocks`, one JSON per
request: hit, per group (manager, spec, block size, dcp_shardable, eagle) the
block id the table resolves each hash to (-1 missing, -2-id if the block's own
hash disagrees), block hashes, prompt ids, and a Counter of call stacks of
every un-cache (`BlockPool._remove_cached_block_hashes`).
`../2026-10-15-c4-timeline/launch_pdump.sh` (holds 2335444, 2336722/23;
queued behind a maintenance reservation).
