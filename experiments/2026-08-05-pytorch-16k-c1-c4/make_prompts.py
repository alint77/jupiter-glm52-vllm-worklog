"""Build 16 unique PyTorch code-generation prompts of exactly 16,384 tokens.

Each prompt is a distinct slice of real PyTorch source from this checkout
followed by a distinct implementation task, so no two prompts share a prefix
and the server's prefix cache cannot shorten any prefill.
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "prompts.jsonl"
MODEL = REPO.parent / "models" / "GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
TARGET_TOKENS = 16384
NUM_PROMPTS = 16

TASKS = [
    "Implement a `MultiQueryFlashAttention` nn.Module in the same style as the "
    "code above: fp16/bf16 autocast-safe, with a `forward(self, q, k, v, "
    "cu_seqlens, max_seqlen)` signature, an explicit causal mask path, and a "
    "torch.compile-friendly implementation that avoids graph breaks.",
    "Write a custom `torch.autograd.Function` implementing fused "
    "RMSNorm + residual add, with a backward that recomputes the norm rather "
    "than saving it, plus a `torch.library.custom_op` registration and a "
    "gradcheck-based unit test.",
    "Write a `torch.utils.data.Dataset` and collate function for variable-"
    "length tokenized sequences that buckets by length, pads to a multiple of "
    "64, and returns cumulative sequence lengths for a varlen attention "
    "kernel. Include a DistributedSampler-compatible wrapper.",
    "Implement a training loop with bf16 autocast, gradient accumulation, "
    "gradient clipping, a cosine schedule with warmup, and activation "
    "checkpointing on every other transformer block. Show how to make the "
    "optimizer step fused and how to avoid a host sync in the logging path.",
    "Write a Triton kernel for a fused SwiGLU forward pass with the same "
    "tiling conventions as the code above, an autotune configuration list, "
    "and a PyTorch wrapper that falls back to eager on unsupported shapes.",
    "Implement an FSDP2 wrapping policy for a mixture-of-experts transformer "
    "that shards experts separately from dense layers, plus the code needed "
    "to save and load a sharded checkpoint without materializing the full "
    "state dict on rank zero.",
    "Write a DDP communication hook that compresses gradients to bf16, "
    "performs the all-reduce, and decompresses back to fp32, with error "
    "feedback accumulated across steps. Include the registration code and a "
    "numerical-drift test against the uncompressed baseline.",
    "Implement a KV-cache manager class supporting paged storage with a fixed "
    "block size, block reuse across requests with a shared prefix, and "
    "eviction by reference count. Give a `torch.Tensor`-backed implementation "
    "and an allocation stress test.",
    "Write a custom `torch.nn.Module` implementing rotary position embeddings "
    "with a precomputed cos/sin cache, support for partial rotary dimensions, "
    "and an in-place variant that avoids allocating on every forward. Include "
    "a correctness test against a naive reference.",
    "Implement a quantization-aware linear layer that stores int4 weights "
    "packed two-per-byte with per-group fp16 scales, dequantizes on the fly "
    "in a Triton matmul, and exposes a `from_float` classmethod. Include the "
    "packing and unpacking utilities and a round-trip test.",
    "Write a profiler harness that wraps a model forward with "
    "`torch.profiler`, emits a Chrome trace, and post-processes it into a "
    "per-kernel table of total time, call count and mean duration. Handle "
    "CUDA-graph-replayed kernels, which carry no Python stack.",
    "Implement a speculative decoding loop in PyTorch: a draft model proposes "
    "k tokens, the target verifies them in one batched forward, and rejected "
    "positions are resampled from the adjusted distribution. Show the exact "
    "acceptance test and prove it preserves the target distribution.",
    "Write a memory-efficient attention implementation using "
    "`torch.nn.functional.scaled_dot_product_attention` with a sliding-window "
    "mask built lazily per batch, plus a fallback chunked implementation for "
    "sequence lengths that exceed available memory.",
    "Implement a custom learning-rate scheduler that supports per-parameter-"
    "group multipliers, linear warmup, cosine decay with restarts, and exact "
    "state save/restore across a preemption. Include a test that a "
    "checkpoint-resume produces a bit-identical LR trajectory.",
    "Write a tensor-parallel column-parallel and row-parallel linear pair "
    "using `torch.distributed`, including the async all-reduce overlap with "
    "the following layer's compute, and a correctness check against the "
    "single-GPU reference under a fixed seed.",
    "Implement an expert-parallel MoE layer: top-k routing, a permutation to "
    "group tokens by expert, an all-to-all dispatch, the grouped expert GEMM, "
    "and the inverse all-to-all combine. Include the load-balancing auxiliary "
    "loss and a single-process test that fakes the collectives.",
]

HEADER = (
    "You are working in a large PyTorch codebase. Below is an excerpt of "
    "existing source for context.\n\n=== BEGIN CONTEXT ===\n"
)
FOOTER = "\n=== END CONTEXT ===\n\nTask: "


def source_pool() -> list[str]:
    roots = [
        REPO / "vllm" / "model_executor" / "layers",
        REPO / "vllm" / "model_executor" / "models",
        REPO / "vllm" / "attention",
        REPO / "vllm" / "v1" / "worker",
        REPO / "vllm" / "distributed",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")))
    texts = []
    for path in files:
        text = path.read_text(errors="ignore")
        if len(text) > 4000:
            texts.append(f"# file: {path.relative_to(REPO)}\n{text}")
    return texts


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    pool = source_pool()
    rng = random.Random(0)
    rng.shuffle(pool)

    records = []
    cursor = 0
    for index in range(NUM_PROMPTS):
        task = TASKS[index]
        tail = FOOTER + task
        budget = TARGET_TOKENS - len(tokenizer.encode(HEADER + tail))

        # Take distinct files per prompt so no two prompts share a prefix.
        chunk_parts = []
        while True:
            chunk_parts.append(pool[cursor % len(pool)])
            cursor += 1
            body = "\n\n".join(chunk_parts)
            if len(tokenizer.encode(body)) > budget + 512:
                break
        # Decoding a token slice does not round-trip exactly, so bisect on the
        # slice length against the count the server will actually see.
        all_ids = tokenizer.encode(body)

        def total(n: int) -> int:
            return len(tokenizer.encode(HEADER + tokenizer.decode(all_ids[:n]) + tail))

        low, high = 0, min(len(all_ids), budget + 512)
        while low < high:
            mid = (low + high + 1) // 2
            if total(mid) <= TARGET_TOKENS:
                low = mid
            else:
                high = mid - 1
        prompt = HEADER + tokenizer.decode(all_ids[:low]) + tail
        count = total(low)
        if count < TARGET_TOKENS - 2:
            raise SystemExit(f"prompt {index} undershot: {count}")

        records.append({"prompt": prompt, "task_index": index, "tokens": count})

    OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    counts = {r["tokens"] for r in records}
    prefixes = {r["prompt"][:200] for r in records}
    print(f"wrote {len(records)} prompts, token counts {counts}")
    print(f"distinct 200-char prefixes: {len(prefixes)}")


if __name__ == "__main__":
    main()
