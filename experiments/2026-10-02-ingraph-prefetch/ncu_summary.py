"""Key ncu metrics and stall reasons per profiled kernel: ncu_summary.py <raw.csv>"""
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr, data = rows[0], rows[2:]
def get(d, name):
    try: return float(d[hdr.index(name)].replace(",", ""))
    except (ValueError, IndexError): return float("nan")
keys = [("gpu__time_duration.sum", "time", 1),
        ("dram__bytes.sum.per_second", "DRAM TB/s", 1),
        ("dram__cycles_active.avg.pct_of_peak_sustained_elapsed", "DRAM active %", 1),
        ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "tensor active %", 1),
        ("sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", "ALU %", 1),
        ("sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", "FMA %", 1),
        ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue active %", 1),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "occupancy %", 1),
        ("smsp__warps_eligible.avg.per_cycle_active", "eligible warps", 1),
        ("gpc__cycles_elapsed.avg.per_second", "SM clock GHz", 1)]
for i, d in enumerate(data):
    print(f"kernel {i}: " + ", ".join(f"{lab} {get(d, k) * sc:.3g}" for k, lab, sc in keys))
    st = [(h.split("issue_stalled_")[1].split("_per_issue")[0], get(d, h)) for h in hdr
          if h.startswith("smsp__average_warps_issue_stalled_") and h.endswith("_per_issue_active.ratio")]
    st = sorted(((n, v) for n, v in st if v == v and v > 0.05), key=lambda kv: -kv[1])
    print("   stalls/issue: " + ", ".join(f"{n} {v:.2f}" for n, v in st))
