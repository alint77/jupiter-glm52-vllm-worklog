#!/usr/bin/env python3
"""Self-contained HTML timeline of a MiMo-V2.6 decode step and one MoE layer.

    make_timeline_html.py timeline-2092055.json --out decode-timeline-2092055.html

One file, no external scripts or fonts: inline SVG drawn by a small script
from the embedded data, hover for details. Three views:
  1. the whole step on a millisecond axis (the GPU stream, layer by layer);
  2. one average sliding-window MoE layer on a microsecond axis, kernel by
     kernel in launch order (a waterfall), with byte floors and the all-reduce
     split into transfer and waiting;
  3. the step budget by part with floors (the "remaining" chart's numbers).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_remaining import ROWS  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiMo decode timeline</title>
<style>
:root {
  --bg: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #8a8984; --grid: #e4e3df;
  --bw: #2a78d6; --bw-dark: #1f5aa3; --lat: #b9b8ad; --comm: #7d6bc4; --wait: #eb6834;
  --attn: #eef3fb; --moe: #fbf1ea; --floor: rgba(11, 11, 11, 0.55); --full: #0f8a7e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --ink: #f1f0ec; --ink2: #c4c3bd; --muted: #8f8e88; --grid: #2e2e2b;
    --bw: #4b93e6; --bw-dark: #7fb2ee; --lat: #6c6b64; --comm: #9d8ee0; --wait: #f07f4f;
    --attn: #1d2430; --moe: #2c221c; --floor: rgba(241, 240, 236, 0.7); --full: #3fbfae;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 4px; }
p.sub, p.note { color: var(--ink2); margin: 0 0 10px; }
p.note { font-size: 12.5px; color: var(--muted); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { fill: var(--ink2); font-size: 11px; }
svg .label { fill: var(--ink); font-size: 12px; }
svg .axis line { stroke: var(--grid); }
svg .hit:hover { filter: brightness(1.15); cursor: default; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 6px 0 2px; font-size: 12.5px; color: var(--ink2); }
.legend span::before { content: ""; display: inline-block; width: 11px; height: 11px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; background: var(--c); }
.legend span.line::before { height: 2px; vertical-align: 3px; }
#tip { position: fixed; pointer-events: none; background: var(--bg); color: var(--ink);
  border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px; font-size: 12.5px;
  box-shadow: 0 4px 16px rgba(0,0,0,.18); max-width: 360px; display: none; z-index: 10; }
#tip b { display: block; margin-bottom: 2px; }
table { border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }
td, th { padding: 3px 10px 3px 0; text-align: right; }
td:first-child, th:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
</style>
</head>
<body>
<main>
<h1>MiMo-V2.6 decode step, kernel by kernel</h1>
<p class="sub">One GH200 of four (TP4 / EP4), 8-token DFlash verify, short context, one-kernel tiered MoE on.
Trace __TRACE__: __STEPS__ steps x __RANKS__ GPUs; layer view aligned over __SAMPLES__ layer instances.
Hover anything for details.</p>
<div class="legend">
  <span style="--c: var(--bw)">bandwidth-bound (bar) with its byte floor (dark line + shade)</span>
  <span style="--c: var(--lat)">small, latency-bound kernel</span>
  <span style="--c: var(--comm)">all-reduce transfer</span>
  <span style="--c: var(--wait)">waiting for the slowest GPU / idle</span>
</div>

<h2>1. One decode step</h2>
<p class="sub">The GPU stream over one step (median over steps and GPUs): the verify graph layer by layer,
then the logits and the DFlash drafter. Host-side kernels (sampling, input prep) fall in the gaps.</p>
<svg id="step"></svg>

<h2>2. One average MoE layer (sliding-window attention; 60 of 70 layers)</h2>
<p class="sub">Every kernel of the layer in launch order on one stream, at its mean start and duration.
Shaded background: attention block, then MoE block.</p>
<svg id="layer"></svg>
<p class="note">The last all-reduce is split into its transfer (the fastest GPU's duration for that step and
layer) and the mean wait for the slowest GPU; that wait is MoE imbalance across GPUs. MoE floors are
E[max(hot bytes / HBM, cold bytes / C2C)] from the expert counts on these prompts, split by bytes (w13 is 2/3).</p>

<h2>3. Where the step's time goes</h2>
<p class="sub">Each part summed over the step, against its floor where one exists (ms per step).</p>
<svg id="budget"></svg>

<div id="tip"></div>
</main>
<script>
const D = __DATA__;
const ROWS = __ROWS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");
function el(parent, name, attrs, text) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}
function hover(e, html) {
  e.classList.add("hit");
  e.addEventListener("mousemove", ev => {
    tip.innerHTML = html; tip.style.display = "block";
    const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px"; tip.style.top = (ev.clientY + 14) + "px";
  });
  e.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}
function axis(svg, x0, x1, y0, y1, tMax, step, fmt) {
  const g = el(svg, "g", {class: "axis"});
  for (let t = 0; t <= tMax + 1e-9; t += step) {
    const x = x0 + (x1 - x0) * t / tMax;
    el(g, "line", {x1: x, x2: x, y1: y0, y2: y1});
    el(g, "text", {x: x, y: y1 + 14, "text-anchor": "middle"}, fmt(t));
  }
}
const fmtUs = v => v.toFixed(1) + " µs";
const kindColor = k => css({bw: "--bw", lat: "--lat", comm: "--comm", wait: "--wait"}[k]);

// ---------------------------------------------------------------- 1. step
(function () {
  const svg = document.getElementById("step");
  const W = 1150, L = 150, R = 20, H = 150;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const P = D.phases_us, T = P.period.end;
  const X = t => L + (W - L - R) * t / T;
  axis(svg, L, W - R, 18, 104, T / 1000, 2, v => v + " ms");
  const rows = [["GPU stream", 28], ["", 0]];
  el(svg, "text", {x: L - 10, y: 50, "text-anchor": "end", class: "label"}, "GPU stream");
  const kinds = {"dense": css("--bw-dark"), "sliding MoE": css("--bw"), "full attention MoE": css("--full")};
  const spans = D.layer_spans_us;
  const layerIds = Object.keys(spans).map(Number).sort((a, b) => a - b);
  // idle before the graph
  const idle = (a, b, why) => {
    if (b - a < 5) return;
    const r = el(svg, "rect", {x: X(a), y: 34, width: Math.max(1, X(b) - X(a)), height: 28, fill: css("--wait"), opacity: 0.75});
    hover(r, `<b>no kernel on the stream</b>${((b - a) / 1000).toFixed(2)} ms, ${why}`);
  };
  idle(0, P.target.start, "step start: input prep and launch");
  for (const k of layerIds) {
    const [a, b] = spans[k];
    const kind = D.layer_kind[k] || "sliding MoE";
    const r = el(svg, "rect", {x: X(a), y: 34, width: Math.max(1, X(b) - X(a) - 0.6), height: 28,
                               fill: kinds[kind], opacity: k % 2 ? 0.85 : 1});
    hover(r, `<b>layer ${k}</b>${kind}<br>${((b - a)).toFixed(0)} µs, starts ${(a / 1000).toFixed(2)} ms`);
  }
  const phase = (name, text, color) => {
    const p = P[name]; if (!p) return;
    const r = el(svg, "rect", {x: X(p.start), y: 34, width: Math.max(1, X(p.end) - X(p.start)), height: 28, fill: color});
    hover(r, `<b>${text}</b>span ${((p.end - p.start) / 1000).toFixed(2)} ms, kernels busy ${(p.busy / 1000).toFixed(2)} ms`);
    return p;
  };
  phase("logits", "logits: lm_head + vocab all-gather", css("--bw-dark"));
  phase("draft", "DFlash drafter", css("--lat"));
  idle(P.target.end, P.logits ? P.logits.start : P.target.end, "between verify graph and logits");
  // annotations
  const ann = (a, b, text) => {
    el(svg, "line", {x1: X(a), x2: X(b), y1: 72, y2: 72, stroke: css("--muted")});
    el(svg, "text", {x: (X(a) + X(b)) / 2, y: 86, "text-anchor": "middle"}, text);
  };
  ann(P.target.start, P.target.end, `verify graph: 70 layers, ${((P.target.end - P.target.start) / 1000).toFixed(1)} ms`);
  ann(P.logits.start, P.draft.end, "logits + drafter");
  el(svg, "text", {x: L, y: 138}, "layers: dark blue = dense layer 0, blue = sliding-window MoE, teal = full-attention MoE (every 8th); orange = no kernel running");
})();

// ---------------------------------------------------------------- 2. layer
(function () {
  const svg = document.getElementById("layer");
  const K = D.layer_kernels;
  const rowH = 24, top = 26, L = 220, R = 250, W = 1150;
  const H = top + K.length * rowH + 44;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const last = K[K.length - 1];
  const end = last.start + last.dur;
  const tMax = Math.ceil(end / 20) * 20;
  const X = t => L + (W - L - R) * t / tMax;
  // blocks
  const moeStart = K.find(k => k.name === "router gate (bf16)").start - 3;
  el(svg, "rect", {x: X(0), y: top - 6, width: X(moeStart) - X(0), height: K.length * rowH + 6, fill: css("--attn")});
  el(svg, "rect", {x: X(moeStart), y: top - 6, width: X(end) - X(moeStart), height: K.length * rowH + 6, fill: css("--moe")});
  el(svg, "text", {x: X(moeStart / 2), y: top - 10, "text-anchor": "middle"}, `attention block, ${moeStart.toFixed(0)} µs`);
  el(svg, "text", {x: X((moeStart + end) / 2), y: top - 10, "text-anchor": "middle"}, `MoE block, ${(end - moeStart).toFixed(0)} µs`);
  axis(svg, L, W - R, top - 6, top + K.length * rowH, tMax, 20, v => v + " µs");
  K.forEach((k, i) => {
    const y = top + i * rowH;
    el(svg, "text", {x: L - 10, y: y + 15, "text-anchor": "end", class: "label"}, k.name);
    const tipBase = `<b>${k.name}</b>starts ${fmtUs(k.start)} into the layer<br>mean ${fmtUs(k.dur)}, median ${fmtUs(k.median)}, p10-p90 ${k.p10.toFixed(1)}-${k.p90.toFixed(1)} µs`;
    let right = X(k.start + k.dur);
    let note = `${k.dur.toFixed(1)} µs`;
    if (k.transfer !== undefined) {
      const tr = el(svg, "rect", {x: X(k.start), y: y + 4, width: Math.max(1.5, X(k.start + k.transfer) - X(k.start)), height: rowH - 8, fill: kindColor("comm"), rx: 2});
      const wt = el(svg, "rect", {x: X(k.start + k.transfer), y: y + 4, width: Math.max(1.5, X(k.start + k.dur) - X(k.start + k.transfer)), height: rowH - 8, fill: kindColor("wait"), rx: 2});
      hover(tr, tipBase + `<br>transfer (fastest GPU) ${fmtUs(k.transfer)}`);
      hover(wt, tipBase + `<br>waiting for the slowest GPU: ${fmtUs(k.dur - k.transfer)} mean (mean wait over GPUs ${fmtUs(k.wait)})<br>= this layer's MoE imbalance across GPUs`);
      note = `${k.dur.toFixed(1)} µs: transfer ${k.transfer.toFixed(1)}, waiting ${(k.dur - k.transfer).toFixed(1)}`;
    } else {
      const r = el(svg, "rect", {x: X(k.start), y: y + 4, width: Math.max(1.5, X(k.start + k.dur) - X(k.start)), height: rowH - 8, fill: kindColor(k.kind), rx: 2});
      let t = tipBase;
      if (k.floor) {
        el(svg, "rect", {x: X(k.start), y: y + 4, width: X(k.start + k.floor) - X(k.start), height: rowH - 8, fill: css("--floor"), opacity: 0.25, "pointer-events": "none"});
        el(svg, "line", {x1: X(k.start + k.floor), x2: X(k.start + k.floor), y1: y + 1, y2: y + rowH - 1, stroke: css("--floor"), "stroke-width": 2});
        t += `<br>byte floor ${fmtUs(k.floor)} -> ${(100 * k.floor / k.dur).toFixed(0)}% of SOL, gap ${fmtUs(k.dur - k.floor)}`;
        note += `  (floor ${k.floor.toFixed(0)}, ${(100 * k.floor / k.dur).toFixed(0)}%)`;
      } else if (k.kind === "lat") {
        t += "<br>latency-bound: too small to be limited by bandwidth";
      }
      hover(r, t);
    }
    el(svg, "text", {x: right + 8, y: y + 15}, note);
  });
  // total gaps between kernels
  let busy = K.reduce((s, k) => s + k.dur, 0);
  el(svg, "text", {x: L, y: H - 4}, `layer ${end.toFixed(0)} µs = kernels ${busy.toFixed(0)} µs + gaps ${(end - busy).toFixed(1)} µs; 69 MoE layers ≈ ${(69 * end / 1000).toFixed(1)} ms of the step`);
})();

// ---------------------------------------------------------------- 3. budget
(function () {
  const svg = document.getElementById("budget");
  const rowH = 26, top = 8, L = 280, R = 260, W = 1150;
  const rows = ROWS.slice().sort((a, b) => (b[1] - (b[2] ?? b[1] / 2)) - (a[1] - (a[2] ?? a[1] / 2)));
  const H = top + rows.length * rowH + 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const tMax = 10;
  const X = t => L + (W - L - R) * t / tMax;
  axis(svg, L, W - R, top, top + rows.length * rowH, tMax, 1, v => v + " ms");
  const kc = {bw: "bw", wait: "wait", lat: "lat"};
  rows.forEach(([name, ms, floor, kind, note], i) => {
    const y = top + i * rowH;
    el(svg, "text", {x: L - 10, y: y + 17, "text-anchor": "end", class: "label"}, name);
    const r = el(svg, "rect", {x: X(0), y: y + 5, width: X(ms) - X(0), height: rowH - 10, fill: kindColor(kc[kind]), rx: 2});
    let t = `<b>${name}</b>${ms.toFixed(2)} ms per step`;
    let lab = `${ms.toFixed(2)} ms`;
    if (floor !== null) {
      el(svg, "rect", {x: X(0), y: y + 5, width: X(floor) - X(0), height: rowH - 10, fill: css("--floor"), opacity: 0.25, "pointer-events": "none"});
      el(svg, "line", {x1: X(floor), x2: X(floor), y1: y + 2, y2: y + rowH - 2, stroke: css("--floor"), "stroke-width": 2});
      t += `<br>floor ${floor.toFixed(2)} ms, gap ${(ms - floor).toFixed(2)} ms`;
      lab += `  gap ${(ms - floor).toFixed(2)}`;
    }
    if (note) t += `<br>${note}`;
    hover(r, t);
    el(svg, "text", {x: X(ms) + 8, y: y + 17}, lab + (note ? "  - " + note : ""));
  });
})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    page = (PAGE.replace("__DATA__", json.dumps(data))
            .replace("__ROWS__", json.dumps(ROWS))
            .replace("__TRACE__", Path(data["trace"]).parent.name + "/" + Path(data["trace"]).name)
            .replace("__STEPS__", str(data["steps"]))
            .replace("__RANKS__", str(data["ranks"]))
            .replace("__SAMPLES__", f"{data['layer_sample_count']:,}"))
    args.out.write_text(page)
    print(args.out, f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
