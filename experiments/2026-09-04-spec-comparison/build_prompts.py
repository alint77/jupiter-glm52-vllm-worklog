#!/usr/bin/env python3
"""Build the four-arm speculative decoding comparison corpus.

Sixty Claude-Code-shaped requests: a realistic coding-agent transcript whose
context is padded to ~120K tokens with real vllm sources, ending in either a
code-writing or a prose task. Ten code and ten prose asks, each over three
different context assemblies. The corpus is built once and replayed verbatim
on every arm, so prompt content is controlled and only the speculator varies.

Measured 2026-09-04 against the live server: real CC traffic on this shape
accepts ~2.5 tokens/step at K=7, not the 5.63 the GSM8K protocol reports --
which is why this corpus exists. GSM8K answers the card's question; this one
answers "which speculator should serve Claude Code".
"""

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SOURCES = [
    "vllm/model_executor/model_loader/tiered_moe_kv.py",
    "vllm/model_executor/model_loader/tiered_moe_physical.py",
    "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
    "vllm/v1/attention/backends/flash_attn.py",
    "vllm/model_executor/models/qwen3_dflash2.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/worker/gpu_worker.py",
    "vllm/config/vllm.py",
    "vllm/config/speculative.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/model_executor/model_loader/tiered_moe_planner.py",
]

# Excised prose the "prior assistant turns" quote, so the transcript reads
# like work on this fork rather than filler. Real sentences from the worklog.
TURN_TAILS = [
    "The planner sizes HBM from fixed_hbm_allocations, which holds the main "
    "MLA cache, the indexer cache and nothing for a dense drafter, so it "
    "promotes hot experts into space the cache already needs.",
    "Reserve is the only lever that moves real headroom, and it moves it "
    "one to one; utilisation only changes the declared budget, which is "
    "slack the tiered path ignores since the planner forces the block count.",
    "The acceptance table is the signature that separates an input problem "
    "from a lattice problem: position zero already low means the draft "
    "disagrees from the first token, deeper positions collapsing means the "
    "walk loses the thread.",
    "The draft page geometry is read off a constructed spec rather than "
    "computed by hand, the way the main and indexer specs already are, so "
    "the accounting lands within a percent of two measured runs.",
    "No reserve value converges: the check needs the observed gap under the "
    "tolerance, and both sides move with the reserve, so the tolerance is "
    "what the gap has to beat.",
    "The runtime reserve check is what stopped the fourth launch; the "
    "autotuner completes on all four workers first, so it was never the "
    "cause, only the nearest suspect.",
]

CODE_TASKS = [
    "Write a pytest test for verify_observed_hbm_reserve that covers the "
    "below-reserve rejection and the tolerance path. Use only functions "
    "visible in the context above and show the complete file.",
    "In build_spec_task we pass 0, 1, 1 for the CP parameters because the "
    "drafter's KV is replicated. Add a comment block and a named helper that "
    "returns those constants, then rewrite the call site.",
    "Write a bash function that waits for a vLLM server on localhost:8027 "
    "and prints the last startup milestone from its log every 15 seconds, "
    "like a wait_for_endpoint loop.",
    "Write a dataclass holding one benchmark request result -- ttft, decode "
    "seconds, output tokens, acceptance length -- plus a summary() that "
    "returns tokens per second.",
    "Rewrite the per-position acceptance extraction as a function that takes "
    "a /metrics dump as text and returns position-to-count as a plain dict.",
    "In the launcher sbatch, the two memory knobs are read from the "
    "environment with defaults. Refactor to a single config function with "
    "explicit validation, and show the diff.",
    "Write a Python script that submits four SLURM jobs with distinct "
    "job names from one template sbatch file, printing job ids as they "
    "return, with no shared mutable state.",
    "Add a constructor validation to TieredKVCachePlan that rejects a "
    "non-negative draft cache with a zero layer count, with the exact "
    "exception style used in this codebase.",
    "Write a kernel-shape probe: given cudagraph capture sizes and a "
    "speculative width, print the schedule's per-step token budget like "
    "the 8192-minus-K warning does.",
    "Refactor wait_for_endpoint so the stage string is only reprinted when "
    "it changes, and add an elapsed-minutes counter from squeue's %M.",
]

PROSE_TASKS = [
    "Summarize the memory-accounting bug in the context above in two "
    "paragraphs for a worklog.",
    "Explain to a colleague why raising the planner reserve can never "
    "satisfy the runtime reserve check. Keep it under 300 words.",
    "Compare how DFlash and DSpark treat below-training-width "
    "configurations and what the user should take away when picking K.",
    "Write the status section of an experiment README for this change: "
    "what launched, what failed, what is still open.",
    "A reviewer asks whether the launch failures were flashinfer's fault. "
    "Write your reply, citing the timeline.",
    "Explain what 'acceptance collapse, not long-context attention' means "
    "when decode throughput drops with context, and how you would verify "
    "that claim from server logs.",
    "Draft the commit message body for a patch that budgets a dense "
    "drafter's KV pages in the tiered planner. Explain the failure mode, "
    "the fix, and the cost.",
    "Describe the trade-off between expert residency and KV budget at a "
    "fixed HBM capacity, for someone operating this deployment.",
    "Explain why the branch count of the drafter's nested KV matters for "
    "planning at DCP 1 versus DCP 4.",
    "Write a short design note arguing for or against blocking this "
    "launcher at 400K context rather than probing lower, given the "
    "evidence in the transcript.",
]

SYSTEM = (
    "You are an interactive coding agent working in a large vLLM fork on a "
    "GH200 cluster. You read code carefully, prefer measurements over "
    "inference, and answer with precise, terse technical prose or exact "
    "code as asked. You may assume the user is the repository's operator "
    "and knows vLLM well."
)


def code_fence(text: str) -> str:
    return f"```\n{text}\n```"


def build_messages(rng: random.Random, task: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    srcs = SOURCES[:]
    rng.shuffle(srcs)
    tails = TURN_TAILS[:]
    rng.shuffle(tails)
    # Alternating transcript pairs: a file paste (user), then a prose reply
    # (assistant) quoting worklog lines. Keeps the shape of a real session.
    # Excerpts are capped so trimming lands near the token budget in coarse
    # steps; kv_cache_utils.py alone is ~85K tokens and would overshoot it.
    for path in srcs:
        body = (REPO / path).read_text(errors="replace")[:48_000]
        if len((REPO / path).read_text(errors="replace")) > 48_000:
            body += "\n# ... (truncated)"
        extra = ""
        if rng.random() < 0.5 and tails:
            extra = "\n\n(`git grep` excerpt below is from the same area.)\n" + code_fence(tails[0])
        msgs.append({
            "role": "user",
            "content": f"Here is {path} as it stands:\n{code_fence(body)}{extra}",
        })
        if tails:
            msgs.append({"role": "assistant", "content": tails.pop(rng.randrange(len(tails)))})
    msgs.append({"role": "user", "content": task})
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "corpus.jsonl"))
    ap.add_argument("--approx-tokens", type=int, default=118_000,
                    help="per-prompt target, measured with --tokenizer when given")
    ap.add_argument("--tokenizer", default="",
                    help="checkpoint dir; sizes prompts by real token count")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def prompt_tokens(msgs: list[dict]) -> int:
        if tok is None:
            return sum(len(m["content"]) for m in msgs) * 10 // 34
        return sum(len(tok(m["content"])["input_ids"]) for m in msgs)

    rows = []
    for kind, tasks in (("code", CODE_TASKS), ("prose", PROSE_TASKS)):
        for ti, task in enumerate(tasks):
            for rep in range(3):
                rng = random.Random(1000 * ti + 7919 * rep + (0 if kind == "code" else 1))
                msgs = build_messages(rng, task)
                budget = (args.approx_tokens
                          if tok is not None
                          else args.approx_tokens * 34 // 10)
                while prompt_tokens(msgs) > budget and len(msgs) > 3:
                    msgs = msgs[:1] + msgs[3:]
                rows.append({
                    "id": f"{kind}-{ti}-{rep}",
                    "task_kind": kind,
                    "messages": msgs,
                    "approx_tokens": prompt_tokens(msgs),
                })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sizes = [r["approx_tokens"] for r in rows]
    print(f"wrote {len(rows)} requests to {args.out}")
    print(f"tokens/request: min {min(sizes)}, median "
          f"{sorted(sizes)[len(sizes)//2]}, max {max(sizes)}, "
          f"total {sum(sizes):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
