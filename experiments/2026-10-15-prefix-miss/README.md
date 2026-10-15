# Append-only prefix-cache misses (~35K cap)

**Root cause: the Anthropic frontend merged Claude Code's inline system
reminders into the leading system prompt.** Fixed by cherry-picking upstream
vllm-project/vllm#58754 (fork 144f2f4aee).

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

## Root cause

- serve builds `AnthropicServingMessages` with
  `chat_template=load_chat_template(args.chat_template)`: `None` without
  `--chat-template`. `_detect_merge_inline_system(None)` returns True (the
  conservative default); GLM-5.3's real `chat_template.jinja` returns False.
- With merge on, `_convert_system_message` drops Claude Code's mid-conversation
  `role: "system"` messages (e.g. "The task tools haven't been used
  recently...") and appends their text to the leading system block.
- GLM renders tools + system first (~35K tokens). Every turn that adds a
  reminder grows that block, so everything after ~35K shifts: the cache hit
  caps at the system block. Turns without a new reminder (#112 -> #113) hit
  99.8%.
- Proof (`inproc_diverge.py`, in-process render, the server's exact path):
  `MERGE_MODE=old_server` reproduces the server's prompt lengths to the token
  (68,738 / 76,334 / 155,927 / 156,492 / 240,176 / 240,430 / 241,223) and
  diverges at 37,702 (#113 -> #114), 35,394 (#16 -> #17), 36,591 (#63 -> #64):
  the server's hits are these floored to the 256-token block, minus the
  block MTP holds back. `MERGE_MODE=template` gives append-only prompts 9-23
  tokens longer (the reminders stay in place).

## Fix

Upstream #58754 (merged 2026-09-25, "Detect Anthropic inline-system merge
against the resolved chat template"): `_should_merge_inline_system` probes the
template the renderer actually applies (`resolve_chat_template`, tools None
and []) and warns at startup if it merges. Cherry-picked as fork 144f2f4aee
(conflicts: only the probe and its call site; the fork's render server has no
merge flag). Prod launchers serve from the main tree and pass no
`--chat-template`, so a server restart picks it up; no launcher change.

- Unit: `tests/entrypoints/anthropic/test_anthropic_messages_conversion.py`
  + `test_chat_error.py` + `test_completion_error.py`: 98 passed.
- CPU, real GLM tokenizer, server's probe (`MERGE_MODE=server
  inproc_diverge.py 112,113,114 16,17 63,64`): merge False; every pair's
  common prefix is the whole previous prompt (#112 240,198 -> #113 240,452 ->
  #114 241,246; #16 68,747 -> #17 76,345; #63 155,944 -> #64 156,510).
- Served (`launch_fixed.sh`, worktree pf5 = pf3 + fix, holds 2336780-82,
  no merge warning in any server log): see below.

## Served result

c=1 replay, fresh server (`sliced-prof/mfix-pfx-2336780/timings.jsonl`):

| request | prompt | hit before | hit after |
|---|---|---|---|
| #112 (cold) | 240,198 | 0 | 0 |
| #113 | 240,452 | 239,872 | 239,872 (99.8%) |
| #114 | 241,246 | 37,376 | 240,128 (99.5%), 0.49 s |
| #16 / #17 | 68,747 / 76,345 | 35,072 / 35,072 | 68,352 / 76,032 |
| #63 / #64 | 155,944 / 156,510 | 36,096 / 36,352 | 155,648 / 156,160 |
| #160 / #161 | 294,969 / 301,784 | 24,320 / 22,784 | 24,320 / 22,784 (tool list changed) |

#16 now hits 68K from #112's blocks (earlier turns are prefixes of later
ones).

c=4 agentic timeline, 600 s, no profiler (`timeline_run.py`, analyzed with
`../2026-10-15-c4-timeline/ab_srpt.py`; before = pf3 holds 2334591/90 on the
same code without the fix):

| arm | turns | out tok/s | hit / prompt | TTFT mean / p50 / p90 (s) | follow-up mean / p90 |
|---|---|---|---|---|---|
| FCFS before | 69 | 97.3 | 75.9% | 9.10 / 0.58 / 33.8 | 6.80 / 30.4 |
| SRPT before | 64 | 89.7 | 75.5% | 9.29 / 0.69 / 39.0 | 6.38 / 34.7 |
| **FCFS fix** | 146 | 229.1 | 95.1% | 2.85 / 0.39 / 1.47 | 1.92 / 0.98 |
| **SRPT fix** | 155 | 232.9 | 95.5% | 2.39 / 0.39 / 1.02 | 1.07 / 0.90 |

2.1-2.4x the turns and output tokens in the same 600 s; prefill tokens
computed 3.09M -> 1.55M while serving 2.5x the prompt tokens.
(`ab_srpt.py`'s "prefix hit" uses the queries counter; SRPT re-runs the
lookup on every admission retry, 180M queried vs 33M prompt tokens, so use
hits / prompt tokens.)

Remaining slow turns (> 10 s, FCFS fix), all explained:
- cold first turns of each agent: #63 (156K, 23 s), #111 (237K, 36 s),
  #159 (294K, 77 s);
- second turns queued behind another agent's cold prefill (FCFS): #64 20 s,
  #112 43 s. SRPT removes these (follow-up mean 1.92 -> 1.07 s, one run
  each, different nodes);
- client-side prompt changes (`MERGE_MODE=server inproc_diverge.py`):
  #161 tool list gained `mcp__ide__executeCode`; #133 the git-status
  snapshot in Claude Code's system prompt changed at token 24,733; #195 and
  #162 changed `Reasoning Effort: High` -> `Max`, which GLM's template renders
  as the very first system token (#162 is also a different request, a
  security-monitor side call).

## Ruled out (before the root cause; note the caveat)

- **Server-side prompt** (wrong): `render_diverge.py` on a GPU-less render
  server (`vllm launch render`) showed append-only prompts, but the fork's
  render server never merges inline system messages, unlike `/v1/messages`
  on the real server. The earlier in-process renders used GLM's template
  (merge False) for the same reason.
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

## Probe (worktree pf4 = 877df9b262 + debug, not for commit; not run)

`VLLM_DEBUG_PREFIX_DUMP=<dir>`: at every `get_computed_blocks`, one JSON per
request: hit, per group (manager, spec, block size, dcp_shardable, eagle) the
block id the table resolves each hash to (-1 missing, -2-id if the block's own
hash disagrees), block hashes, prompt ids, and a Counter of call stacks of
every un-cache (`BlockPool._remove_cached_block_hashes`).
`../2026-10-15-c4-timeline/launch_pdump.sh`. Its holds were cancelled
unrun (maintenance) once the frontend cause was found.
