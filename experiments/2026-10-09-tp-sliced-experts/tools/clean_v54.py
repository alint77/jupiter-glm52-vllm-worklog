"""td_v54.cu (unifdef_td.py of td_v53 -DTD_NOAHEAD) -> the served kernel's
code: dead trace hooks, the unused ready list and the disabled inline w13
handoff removed."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
p = Path(sys.argv[1])
src = p.read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


def strip_calls(text, name):
    res, i = [], 0
    pat = re.compile(r"(?m)^[ \t]*" + name + r"\(")
    while True:
        m = pat.search(text, i)
        if not m:
            res.append(text[i:])
            return "".join(res)
        j, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        while j < len(text) and text[j] in " \t;":
            j += 1
        if j < len(text) and text[j] == "\n":
            j += 1
        res.append(text[i:m.start()])
        i = j


sub("  #define TD_T1(v)\n  #define TD_REC(ph, t1)\n  #define TD_KREC(ph)\n", "")
for name in ("TD_T1", "TD_REC", "TD_KREC"):
    src = strip_calls(src, name)
sub("  #define TD_MMA_ASM asm\n", "")
sub("  TD_MMA_ASM(", "  asm(")
sub("  #define TD_GUIDED 0  // max routed w2 units per claim, guided (0: GR1)\n", "")
sub('static_assert(TD_GUIDED == 0, "v39 sizes w2 claims per call, not guided");\n', "")
sub("""  int rl_tail[2];  // per tier: entries appended to rl, zeroed by route_prep
  // per tier, in ready order: {epoch, entry} once the entry's x2 rows are
  // written
  alignas(8) unsigned long long rl[2][MAX_LIST];
""", "")
sub("    ws->rl_tail[0] = ws->rl_tail[1] = 0;\n", "")
a = src.index("__device__ __forceinline__ unsigned long long ld_acquire64(")
b = src.index("__device__ __forceinline__ void fence_proxy_async() {")
src = src[:a] + src[b:]
a = src.index("        if (true) {  // the finisher counts; nothing more here\n")
b = src.index("    } else if (kind == K_R1) {", a)
tail = "      }\n"
src = src[:a] + src[b - len(tail):]
assert "TD_T1" not in src and "TD_KREC" not in src and "rl_tail" not in src
p.write_text(src)
print("ok")
