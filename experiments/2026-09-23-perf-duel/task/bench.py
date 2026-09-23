"""Correctness, time and peak memory for solution.py. Do not edit.

Usage: python bench.py [--shape small|main|large] [--json out.json]
"""

import argparse
import importlib
import json
import math
import statistics
import sys

import torch

import reference

SHAPES = {
    "small": (4096, 2048, 32000),
    "main": (8192, 4096, 151936),
    "large": (16384, 6144, 152576),
}
LOSS_RTOL = 1e-3
GRAD_RTOL = 1e-2
WARMUP = 3
ITERS = 10


def make_inputs(n, h, v, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, h, device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn(v, h, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    t = torch.randint(0, v, (n,), device="cuda", generator=gen)
    t[torch.rand(n, device="cuda", generator=gen) < 0.1] = reference.IGNORE_INDEX
    return x.requires_grad_(), w.requires_grad_(), t


def run_once(fn, x, w, t):
    x.grad = None
    w.grad = None
    loss = fn(x, w, t)
    loss.backward()
    return loss


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


def check(fn, x, w, t):
    loss = run_once(fn, x, w, t)
    ref_loss, ref_gx, ref_gw = reference.fp32_reference(x.detach(), w.detach(), t)
    problems = []
    if loss.dim() != 0 or loss.dtype != torch.float32:
        problems.append(f"loss must be a float32 scalar, got {loss.dtype} {loss.shape}")
    if x.grad is None or w.grad is None:
        problems.append("x.grad / weight.grad not populated")
        return problems, {}
    if x.grad.dtype != torch.bfloat16 or w.grad.dtype != torch.bfloat16:
        problems.append("gradients must be bfloat16")
    errs = {
        "loss": abs(loss.item() - ref_loss.item()) / abs(ref_loss.item()),
        "grad_x": rel_err(x.grad, ref_gx),
        "grad_w": rel_err(w.grad, ref_gw),
    }
    if not math.isfinite(loss.item()) or errs["loss"] > LOSS_RTOL:
        problems.append(f"loss rel err {errs['loss']:.2e}")
    for key in ("grad_x", "grad_w"):
        if not math.isfinite(errs[key]) or errs[key] > GRAD_RTOL:
            problems.append(f"{key} rel err {errs[key]:.2e}")
    return problems, errs


def measure(fn, input_sets):
    for i in range(WARMUP):
        run_once(fn, *input_sets[i % len(input_sets)])
    torch.cuda.synchronize()
    times = []
    for i in range(ITERS):
        x, w, t = input_sets[i % len(input_sets)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(fn, x, w, t)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    x, w, t = input_sets[0]
    x.grad = None
    w.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run_once(fn, x, w, t)
    torch.cuda.synchronize()
    grad_bytes = x.grad.numel() * x.grad.element_size()
    grad_bytes += w.grad.numel() * w.grad.element_size()
    extra = torch.cuda.max_memory_allocated() - base - grad_bytes
    return statistics.median(times), max(extra, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES), action="append")
    parser.add_argument("--solution", default="solution")
    parser.add_argument("--json")
    args = parser.parse_args()
    solution = importlib.import_module(args.solution).fused_linear_cross_entropy
    baseline = reference.baseline_linear_cross_entropy

    results = {}
    for name in args.shape or list(SHAPES):
        n, h, v = SHAPES[name]
        sets = [make_inputs(n, h, v, seed) for seed in (1, 2)]
        problems = []
        errs = {}
        for x, w, t in sets:
            p, errs = check(solution, x, w, t)
            problems += p
        row = {"shape": (n, h, v), "errors": errs, "problems": problems}
        if not problems:
            sol_ms, sol_mem = measure(solution, sets)
            base_ms, base_mem = measure(baseline, sets)
            row |= {
                "ms": sol_ms,
                "extra_gib": sol_mem / 2**30,
                "baseline_ms": base_ms,
                "baseline_extra_gib": base_mem / 2**30,
                "speedup": base_ms / sol_ms,
                "mem_reduction": base_mem / max(sol_mem, 2**20),
            }
        results[name] = row
        status = "FAIL " + "; ".join(problems) if problems else "ok"
        print(f"[{name}] N={n} H={h} V={v}: {status}")
        print(
            "  errors: "
            + ", ".join(f"{k} {e:.2e}" for k, e in errs.items())
        )
        if not problems:
            print(
                f"  time {row['ms']:.2f} ms (baseline {row['baseline_ms']:.2f}, "
                f"speedup {row['speedup']:.2f}x); extra memory "
                f"{row['extra_gib']:.2f} GiB (baseline "
                f"{row['baseline_extra_gib']:.2f}, {row['mem_reduction']:.1f}x less)"
            )
        del sets
        torch.cuda.empty_cache()

    scored = [results[s] for s in ("main", "large") if s in results]
    if all(not r["problems"] for r in results.values()) and scored:
        score = math.exp(
            statistics.mean(math.log(r["speedup"] * r["mem_reduction"]) for r in scored)
        )
        print(f"SCORE (geomean speedup x mem_reduction over main,large): {score:.2f}")
        results["score"] = score
    else:
        print("SCORE: 0 (a shape failed or was not run)")
        results["score"] = 0.0
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0 if results["score"] else 1


if __name__ == "__main__":
    sys.exit(main())
