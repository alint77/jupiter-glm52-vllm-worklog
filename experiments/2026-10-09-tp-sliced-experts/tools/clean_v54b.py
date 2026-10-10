"""Second pass on td_v54.cu: overridable tunables (#ifndef) as in the served
kernel, the dead w13 bar.sync handoff helpers and the XROWS timing probe
removed."""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text()


def sub(old, new):
    global src
    assert src.count(old) == 1, old[:90]
    src = src.replace(old, new)


for name in ("TD_STAGES", "TD_HQ", "TD_ACTK", "TD_SQ", "TD_COLD_CTAS",
             "TD_GR1", "TD_GR1_T", "TD_R0S", "TD_FIN_FENCE"):
    m = re.search(r"(?m)^  #define " + name + r"\b.*\n", src)
    src = (src[:m.start()] + f"#ifndef {name}\n" + m.group(0) + "#endif\n"
           + src[m.end():])
sub("""// timing probe: x rows reserved per stage (copies and reads clamped; results
// invalid when an entry has more tokens)
  #define TD_XROWS MAX_TOK
constexpr int XROWS = TD_XROWS;
""", "constexpr int XROWS = MAX_TOK;\n")
sub("""// w13 completion handoff: consumer warps 1..7 arrive, warp 0 syncs
__device__ __forceinline__ void handoff_arrive() {
  asm volatile("bar.arrive 2, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}
__device__ __forceinline__ void handoff_sync() {
  asm volatile("bar.sync 2, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}
""", "")
p.write_text(src)
print("ok")


def drop_dead_locals(path):
    """Third pass: locals left unused by the removed trace and handoff code."""
    p = Path(path)
    s = p.read_text()
    for old in (
        "  uint64_t td_t1 = 0;\n  (void)td_t1;\n",
        "    const float xs13r =\n"
        "        lane < T ? ws->xs13[lane] : 0.f;  // token lane's x13 scale\n",
        "  int acq = -1;  // (tier, entry) whose ready this warp has acquired\n",
        "  const uint32_t wo0 = (warp * 8) * 512 + lane * 16;\n"
        "  const uint32_t so0 = W_BYTES + (warp * 4) * 128 + 16 * g;\n"
        "  const uint32_t xo0 =\n"
        "      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + "
        "(warp * 16 + tq) * 16;\n",
        "    const Expert* experts = ws->lists[q];\n",
    ):
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, "")
    p.write_text(s)
