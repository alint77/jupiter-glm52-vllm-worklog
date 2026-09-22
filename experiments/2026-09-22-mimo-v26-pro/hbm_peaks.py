"""Per-phase peak HBM per GPU from bench.sbatch's nvidia-smi and phase logs."""

import csv
import sys
from datetime import datetime
from pathlib import Path

FMT = "%Y/%m/%d %H:%M:%S.%f"


def main(result_dir: str) -> None:
    d = Path(result_dir)
    phases = [
        (datetime.strptime(t, FMT), name)
        for t, name in csv.reader(open(d / "phases.csv"))
    ]
    samples = []
    for row in csv.reader(open(d / "hbm.csv")):
        if len(row) < 4:
            continue
        try:
            samples.append(
                (
                    datetime.strptime(row[0].strip(), FMT),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                )
            )
        except ValueError:
            continue
    total = max(s[3] for s in samples)
    print(f"HBM total {total / 1024:.2f} GiB; per phase: max used GiB per GPU (min free)")
    for (start, name), (end, _) in zip(phases, phases[1:]):
        per_gpu: dict[int, int] = {}
        for t, gpu, used, _ in samples:
            if start <= t < end:
                per_gpu[gpu] = max(per_gpu.get(gpu, 0), used)
        if not per_gpu:
            continue
        used = " ".join(f"{per_gpu[g] / 1024:6.2f}" for g in sorted(per_gpu))
        free = (total - max(per_gpu.values())) / 1024
        print(f"  {name:16s} {used}   (min free {free:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
