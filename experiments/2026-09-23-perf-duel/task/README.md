# Task: fused linear + cross-entropy (forward and backward) on H100/GH200

You are a GPU performance engineer. Implement the language-model loss head

    loss = cross_entropy(x @ W.T, target, ignore_index=-100)   # mean over non-ignored rows

as fast and as memory-lean as you can, **including the backward pass** for both
`x` and `W`.

## Why it is hard

At LLM vocabulary sizes the naive version materialises an `N x V` logits tensor
(8192 x 151936 in fp32 is 5 GB), then a softmax of the same size, then a
gradient of the same size. The whole point of this task is to avoid that while
still being fast: the matmul is ~10 TFLOP per call, so a correct but slow
chunked loop is easy and a fast one is not.

## Interface (do not change)

Edit **only** `solution.py`. It must define

```python
def fused_linear_cross_entropy(
    x: torch.Tensor,        # [N, H] bfloat16, CUDA, requires_grad
    weight: torch.Tensor,   # [V, H] bfloat16, CUDA, requires_grad
    target: torch.Tensor,   # [N] int64, values in [0, V) or -100 (ignored)
) -> torch.Tensor:          # scalar float32 loss, differentiable w.r.t. x and weight
```

Calling `loss.backward()` must populate `x.grad` and `weight.grad` (bf16).
The loss is the mean over rows whose target is not -100.

## Rules

* Allowed: PyTorch and Triton (both installed). Nothing else: no
  `liger_kernel`, `cut_cross_entropy`, `apex`, `xformers`, flash-attn, or any
  other kernel library, and no custom CUDA/C++ extensions.
* Do not edit `bench.py` or `reference.py`; the grader uses pristine copies
  with fresh random inputs, and runs your function on several input sets, so
  caching results across calls is detected.
* The GPU you have is `CUDA_VISIBLE_DEVICES` (one H100-class GH200, 96 GB HBM,
  sm_90). Use the Python at `$PY` (see below). Nothing needs installing.

## Grading (run it yourself any time)

    $PY bench.py            # all shapes: correctness, time, peak memory
    $PY bench.py --shape main

Shapes: `small` (N=4096, H=2048, V=32000), `main` (N=8192, H=4096,
V=151936), `large` (N=16384, H=6144, V=152576). About 10% of targets are
-100.

Correctness (must hold on every shape, or the submission scores zero):
* loss relative error < 1e-3 vs an fp32 reference;
* `x.grad` and `weight.grad` relative Frobenius error < 1e-2.

Performance, reported per shape against the PyTorch baseline in
`reference.py` (logits in fp32, `F.cross_entropy`):
* **time**: median forward+backward wall time (CUDA events, warm);
* **peak memory**: extra allocated memory during forward+backward, beyond the
  inputs and the two gradient tensors.

The score is the geometric mean over `main` and `large` of
`speedup x memory_reduction`. A good solution is both faster than the
baseline and uses a small fraction of its extra memory; either one alone is
easy.

## Deliverables

* `solution.py` -- always keep it working; the last passing version counts.
* `NOTES.md` -- your approach, what you tried, and the final `bench.py`
  numbers you measured.

You have a fixed time budget; work autonomously, measure often, and do not
stop at the first correct version.
