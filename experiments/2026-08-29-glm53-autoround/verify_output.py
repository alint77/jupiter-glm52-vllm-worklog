#!/usr/bin/env python3
"""Check the GLM-5.3 AutoRound output against the GLM-5.2 reference recipe.

Compares what was quantized rather than trusting the emitted config: every
tensor family in the 5.2 checkpoint is matched against the 5.3 output, so a
module that should have stayed BF16 but got packed (or the reverse) shows up
as a structural difference rather than a silent quality regression.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REF = "/e/project1/profound/alint77/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(REF))
    parser.add_argument(
        "--expect-mtp",
        action="store_true",
        help="require layer 78 experts to be quantized; off by default because "
        "this run deliberately leaves the MTP head in BF16",
    )
    return parser.parse_args()


def families(index_path: Path) -> tuple[Counter, dict[str, set[int]]]:
    """Bucket tensor names into families, and record which layers are packed."""
    weight_map = json.loads(index_path.read_text())["weight_map"]
    fam: Counter = Counter()
    packed_layers: dict[str, set[int]] = {"quantized": set(), "bf16": set()}
    for name in weight_map:
        layer = re.search(r"layers\.(\d+)\.", name)
        idx = int(layer.group(1)) if layer else -1
        if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
            kind = "quantized"
        elif name.endswith(".weight") or name.endswith(".bias"):
            kind = "bf16"
        else:
            kind = "other"
        if ".mlp.experts." in name:
            group = "routed_expert"
        elif ".mlp.shared_experts." in name:
            group = "shared_expert"
        elif ".mlp.gate" in name:
            group = "router_gate"
        elif ".self_attn." in name:
            group = "attention"
        elif "eh_proj" in name:
            group = "eh_proj"
        elif idx == -1:
            group = "global"
        else:
            group = "other_layer"
        fam[(group, kind)] += 1
        if group == "routed_expert" and kind in packed_layers:
            packed_layers[kind].add(idx)
    return fam, packed_layers


def main() -> None:
    args = parse_args()
    out_idx = args.output / "model.safetensors.index.json"
    ref_idx = args.reference / "model.safetensors.index.json"
    if not out_idx.is_file():
        raise SystemExit(f"no index in {args.output}")

    out_fam, out_layers = families(out_idx)
    ref_fam, ref_layers = families(ref_idx)

    print(f"{'family':>16} {'kind':>10} {'5.3 out':>9} {'5.2 ref':>9}")
    for key in sorted(set(out_fam) | set(ref_fam)):
        group, kind = key
        print(f"{group:>16} {kind:>10} {out_fam.get(key, 0):>9} {ref_fam.get(key, 0):>9}")

    print(f"\nrouted-expert layers quantized:")
    o, r = sorted(out_layers["quantized"]), sorted(ref_layers["quantized"])
    print(f"  5.3 out: {o[0] if o else '-'}..{o[-1] if o else '-'} (n={len(o)})")
    print(f"  5.2 ref: {r[0] if r else '-'}..{r[-1] if r else '-'} (n={len(r)})")

    problems = []
    for group in ("attention", "shared_expert", "router_gate", "eh_proj"):
        n = out_fam.get((group, "quantized"), 0)
        if n:
            problems.append(f"{group} has {n} quantized tensors; recipe keeps it BF16")
    expected = set(range(3, 79)) if args.expect_mtp else set(range(3, 78))
    if set(o) != expected:
        problems.append(
            f"quantized layer set {sorted(set(o) ^ expected)} differs from expected"
        )
    if not out_fam.get(("routed_expert", "quantized")):
        problems.append("no routed experts were quantized at all")

    print("\n" + ("FAIL\n  " + "\n  ".join(problems) if problems
                  else "PASS: structure matches the recipe"))


if __name__ == "__main__":
    main()
