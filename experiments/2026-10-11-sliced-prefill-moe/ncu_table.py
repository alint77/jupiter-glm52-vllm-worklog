"""Summary table of an ncu raw csv (ncu.sh)."""
import csv
import re
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr, units, data = rows[0], rows[1], rows[2:]
col = {h: i for i, h in enumerate(hdr)}
SCALE = {"ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3, "byte": 1e-6,
         "Kbyte": 1e-3, "Mbyte": 1, "Gbyte": 1e3}


def g(r, m):
    v = r[col[m]].replace(",", "")
    try:
        return float(v) * SCALE.get(units[col[m]], 1)
    except ValueError:
        return float("nan")
print(f"{'kernel':22s} {'grid':>7s} {'regs':>4s} {'us':>7s} {'tensor%':>7s} {'dram%':>6s} {'rdMB':>7s} {'wrMB':>7s} {'issue%':>6s} {'warps%':>6s} {'smact%':>6s} {'tens/el%':>8s} {'GHz':>5s}")
for r in data:
    n = r[col["Kernel Name"]]
    m = re.search(r"(\w+_kernel)(<\D*(\d+))?", n)
    name = m[1] + (" " + m[3] if m[3] else "")
    print(f"{name:22s} {g(r,'launch__grid_size'):7.0f} {g(r,'launch__registers_per_thread'):4.0f}"
          f" {g(r,'gpu__time_duration.sum'):7.1f} {g(r,'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'):7.1f}"
          f" {g(r,'dram__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f} {g(r,'dram__bytes_read.sum'):7.1f}"
          f" {g(r,'dram__bytes_write.sum'):7.1f} {g(r,'smsp__issue_active.avg.pct_of_peak_sustained_active'):6.1f}"
          f" {g(r,'sm__warps_active.avg.pct_of_peak_sustained_active'):6.1f}"
          f" {100 * g(r,'sm__cycles_active.avg') / g(r,'gpc__cycles_elapsed.max'):6.1f}"
          f" {g(r,'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'):8.1f}"
          f" {g(r,'gpc__cycles_elapsed.max') / g(r,'gpu__time_duration.sum') / 1e3:5.2f}")
