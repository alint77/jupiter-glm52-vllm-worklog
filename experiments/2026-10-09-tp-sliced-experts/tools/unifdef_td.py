"""Resolve a td_vNN.cu experiment kernel to one configuration: every
#if/#ifdef/#ifndef/#elif whose condition only involves TD_* macros is
evaluated against the macro set the file itself produces for the given -D
flags (gcc -E -dM), and dropped; other conditionals stay. TD_V(...) trace
statements and empty boolean #defines of TD_* flags are removed; value macros
(TD_STAGES 4, ...) are kept.
  unifdef_td.py in.cu out.cu [-DTD_X[=v] ...] [--keep TD_A,TD_B]"""
import re
import sys
from pathlib import Path

src_path, out_path, *flags = sys.argv[1:]
keep = set()
if "--keep" in flags:
    i = flags.index("--keep")
    keep = set(flags[i + 1].split(","))
    del flags[i:i + 2]
src = Path(src_path).read_text()

# macros as the preprocessor sees them, line by line: command-line flags, then
# every active #define / #undef of a TD_* macro
macros = {}
for f in flags:
    k, _, v = f[2:].partition("=")
    macros[k] = v
fn_macros = set()


def td_only(expr):
    ids = set(re.findall(r"[A-Za-z_]\w*", expr)) - {"defined"}
    return ids and all(i.startswith("TD_") and i not in keep for i in ids)


def evaluate(expr):
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in macros
               or m.group(1) in fn_macros else "0", expr)
    e = re.sub(r"defined\s+(\w+)", lambda m: "1" if m.group(1) in macros else "0", e)
    e = re.sub(r"TD_\w+", lambda m: macros.get(m.group(0), "0") or "1", e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = re.sub(r"//.*", "", e)
    return bool(eval(e))


out = []
# stack entries: (resolved, emitting_parent, branch_taken, current_active)
stack = []


def active():
    return all(s[3] for s in stack)


lines = src.splitlines()
for line in lines:
    s = line.strip()
    m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
    if not m:
        if active():
            out.append(line)
            d = re.match(r"#\s*define\s+(TD_\w+)(\()?(?:[ \t]+(.*))?$", s)
            if d:
                if d.group(2):
                    fn_macros.add(d.group(1))
                else:
                    macros[d.group(1)] = re.sub(r"//.*", "", d.group(3) or "").strip()
            u = re.match(r"#\s*undef\s+(TD_\w+)", s)
            if u:
                macros.pop(u.group(1), None)
        continue
    kind, rest = m.group(1), m.group(2).split("//")[0].strip()
    if kind in ("ifdef", "ifndef", "if"):
        expr = (f"defined({rest})" if kind == "ifdef" else
                f"!defined({rest})" if kind == "ifndef" else rest)
        if td_only(expr):
            v = evaluate(expr)
            stack.append([True, None, v, v])
        else:
            stack.append([False, None, True, True])
            if active():
                out.append(line)
    elif kind == "elif":
        top = stack[-1]
        if top[0] and td_only(rest):
            v = (not top[2]) and evaluate(rest)
            top[2] = top[2] or v
            top[3] = v
        else:
            assert not top[0], f"mixed #elif: {line}"
            if active() or True:
                out.append(line)
    elif kind == "else":
        top = stack[-1]
        if top[0]:
            top[3] = not top[2]
            top[2] = True
        else:
            stack[-1][3] = True
            if all(x[3] for x in stack[:-1]):
                out.append(line)
    else:  # endif
        top = stack.pop()
        if not top[0] and active():
            out.append(line)
text = "\n".join(out) + "\n"


def strip_calls(text, name):
    res, i = [], 0
    pat = re.compile(r"(?m)^[ \t]*" + name + r"\(")
    while True:
        m = pat.search(text, i)
        if not m:
            res.append(text[i:])
            break
        j, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        k = j
        while k < len(text) and text[k] in " \t":
            k += 1
        if k < len(text) and text[k] == "\n":
            k += 1
        res.append(text[i:m.start()])
        i = k
    return "".join(res)


if "TD_V" not in macros or True:
    text = strip_calls(text, "TD_V")
text = re.sub(r"(?m)^[ \t]*#define TD_V\(.*\n", "", text)
# empty object-like TD_* macros (flags, or hooks that expand to nothing in
# this configuration): drop the define and every use
empty = {k for k, v in macros.items() if v == "" and k not in keep}
text = re.sub(r"(?m)^[ \t]*#\s*define (TD_\w+)[ \t]*(//.*)?\n",
              lambda m: "" if m.group(1) in empty else m.group(0), text)
out2 = []
for l in text.split("\n"):
    if l.lstrip().startswith("#"):
        out2.append(l)
        continue
    n = re.sub(r"\b(" + "|".join(sorted(empty)) + r")\b ?", "", l) if empty else l
    if n.strip() or not l.strip():
        out2.append(n)
text = "\n".join(out2)
text = re.sub(r"\n{3,}", "\n\n", text)
Path(out_path).write_text(text)
left = sorted(set(re.findall(r"TD_\w+", text)))
print("macros left:", " ".join(left))
