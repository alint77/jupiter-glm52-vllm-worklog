"""Tiered MoE on a 4x GH200 node, explained with MiMo-V2.6-Pro.

Render one scene:   manim -qh tiered_moe.py S1Hardware
Render all + join:  bash render.sh [l|m|h]

Numbers come from the JSC JUPITER configuration page, the MiMo-V2.6-Pro
checkpoint, data.json / step.json (real routing traces and the production
placement profile, see prep_data.py and prep_step.py), the per-GPU timeline of
one real decode layer (figs/layer-ranks.png in the write-up) and measured
serving results.

Pacing rule: one point at a time. A visual change plays, then the scene holds
long enough to read it; a caption is never introduced in the same beat as the
visual it explains.
"""

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Arrow,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    config,
    linear,
)

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text())
STEP = json.loads((HERE / "step.json").read_text())

BG = "#101114"
INK = "#ECEBE6"
INK2 = "#A9A8A2"
MUTED = "#6E6D68"
CELL = "#4A4A48"
HBM = "#3D8BEA"  # VRAM / hot
DDR = "#F07A3A"  # Grace RAM / cold
C2C = "#E9C46A"
NVL = "#7BC8A4"
WAIT = "#1FAF7A"
OTHER = "#5C5B57"
BAD = "#E05B5B"
GOOD = "#7BC8A4"
FONT = "DejaVu Sans"
MAX_W = 13.4

config.background_color = BG


def T(text, size=28, color=INK, weight="NORMAL"):
    # Rendered large and scaled down: small Pango sizes kern unevenly. Pango
    # wraps lines past a fixed width, so use the largest factor that keeps the
    # text on one line.
    for f in (4, 2, 1):
        t = Text(text, font=FONT, font_size=size * f, color=color, weight=weight,
                 disable_ligatures=True)
        ref = Text("Ag", font=FONT, font_size=size * f, weight=weight)
        if t.height < 1.6 * ref.height:
            break
    t.scale(1 / f)
    if t.width > MAX_W:
        t.scale_to_fit_width(MAX_W)
    return t


def chars(m):
    if isinstance(m, Text):
        return len(m.original_text)
    return sum(chars(s) for s in m.submobjects)


def read(scene, *mobs, extra=0):
    """Hold for the time it takes to read the text just shown."""
    n = sum(chars(m) for m in mobs)
    scene.wait(max(1.1, min(4.0, 0.6 + 0.038 * n)) + extra)


# Holds after a visual change, sized to how much there is to take in.
HOLD = {"quick": 0.35, "look": 0.7, "long": 1.0, "study": 2.3}


def pause(scene, kind):
    scene.wait(HOLD[kind])


def show(scene, mob, extra=0.0, shift=UP * 0.15):
    scene.play(FadeIn(mob, shift=shift), run_time=0.6)
    read(scene, mob, extra=extra)


def title(scene, text, sub=None):
    t = T(text, 40, weight="BOLD").to_edge(UP, buff=0.35)
    group = VGroup(t)
    if sub:
        group.add(T(sub, 24, INK2).next_to(t, DOWN, buff=0.15))
    scene.play(FadeIn(group, shift=DOWN * 0.2), run_time=0.7)
    scene.wait(1.3 if sub else 0.9)
    return group


def box(w, h, color, fill=0.16):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color,
                            stroke_width=3, fill_color=color, fill_opacity=fill)


def clear(scene, run_time=0.5):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)
    scene.wait(0.2)


def chip_part(name, lines, color, w=2.3, h=1.55, big=22, small=17):
    g = VGroup(box(w, h, color), T(name, big, weight="BOLD"), *[T(x, small, INK2) for x in lines])
    g[1:].arrange(DOWN, buff=0.08).move_to(g[0])
    return g


def grace(w=2.3, h=1.55, big=22, small=17):
    return chip_part("Grace CPU", ["72 cores", "120 GB LPDDR5X", "512 GB/s"], DDR, w, h, big, small)


def hopper(w=2.3, h=1.55, big=22, small=17):
    return chip_part("Hopper GPU", ["132 SMs", "96 GB HBM3", "4 TB/s"], HBM, w, h, big, small)


# --------------------------------------------------------------------------
class S1Hardware(Scene):
    def construct(self):
        title(self, "The machine: one JUPITER Booster node", "4x NVIDIA GH200 Grace-Hopper superchips")

        g, h = grace(), hopper()
        h.next_to(g, RIGHT, buff=2.4)
        one = VGroup(g, h).scale(1.2).move_to(DOWN * 0.2)
        self.play(FadeIn(g, shift=RIGHT * 0.2), FadeIn(h, shift=LEFT * 0.2), run_time=1.0)
        pause(self, "long")
        link = Line(g.get_right(), h.get_left(), color=C2C, stroke_width=12)
        lab = T("NVLink-C2C", 24, C2C, weight="BOLD").next_to(link, UP, buff=0.15)
        self.play(Create(link), FadeIn(lab), run_time=1.0)
        pause(self, "look")
        bw = T("900 GB/s total, 450 GB/s each way", 26, C2C).next_to(one, DOWN, buff=0.5)
        show(self, bw)
        note = T("the GPU reads Grace memory directly, ~7x faster than PCIe Gen5 x16", 22, INK2).next_to(bw, DOWN, buff=0.25)
        show(self, note)
        self.play(FadeOut(VGroup(g, h, link, lab, bw, note)), run_time=0.7)

        # Four superchips as columns: Grace on top, Hopper below.
        cols = VGroup()
        for i in range(4):
            gg, hh = grace(1.95, 1.3, 19, 14), hopper(1.95, 1.3, 19, 14)
            hh.next_to(gg, DOWN, buff=0.75)
            ln = Line(gg.get_bottom(), hh.get_top(), color=C2C, stroke_width=9)
            cl = T("C2C", 16, C2C, weight="BOLD").next_to(ln, RIGHT, buff=0.1)
            col = VGroup(gg, hh, ln, cl)
            col.gpu, col.cpu = hh, gg
            cols.add(col)
        cols.arrange(RIGHT, buff=0.75).move_to(UP * 0.35)
        names = VGroup(*[T(f"superchip {i}", 17, INK2).next_to(c, UP, buff=0.12) for i, c in enumerate(cols)])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.25), run_time=1.8)
        self.play(FadeIn(names), run_time=0.6)
        pause(self, "long")

        arcs = VGroup()
        for i in range(4):
            for j in range(i + 1, 4):
                d = j - i
                a = cols[i].gpu.get_bottom() + RIGHT * (0.18 * (j - 2.5))
                b = cols[j].gpu.get_bottom() + LEFT * (0.18 * (1.5 - i))
                arcs.add(ArcBetweenPoints(a, b, angle=PI * (0.35 + 0.12 * d), color=NVL, stroke_width=3))
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=1.8)
        pause(self, "look")
        nv = T("NVLink 4: every GPU pair, 150 GB/s each way", 24, NVL).to_edge(DOWN, buff=0.35)
        show(self, nv)
        self.play(FadeOut(arcs), FadeOut(nv), run_time=0.7)

        node = VGroup(cols, names)
        self.play(node.animate.scale(0.55).to_edge(LEFT, buff=0.3).shift(DOWN * 0.2), run_time=1.2)
        rows = VGroup(
            T("Per node", 28, weight="BOLD"),
            T("384 GB HBM3  (4 x 96)", 24, HBM),
            T("480 GB LPDDR5X  (4 x 120)", 24, DDR),
            T("~630 of ~990 TFLOPS BF16 (power-limited)", 22, INK2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(RIGHT, buff=0.5).shift(UP * 1.7)
        self.play(FadeIn(rows[0]), run_time=0.5)
        for r in rows[1:]:
            self.play(FadeIn(r, shift=LEFT * 0.2), run_time=0.6)
            read(self, r, extra=-0.8)

        # (name, spec GB/s, achievable GB/s or None, colour)
        ladder = [("HBM -> GPU", 4000, 3600, HBM), ("Grace RAM -> GPU (C2C)", 450, 420, C2C),
                  ("GPU <-> GPU (NVLink)", 150, None, NVL), ("Grace <-> Grace", 100, 90, DDR),
                  ("node <-> node (IB)", 25, None, BAD)]
        bars = VGroup()
        for name, spec, got, col in ladder:
            W = 4.3 / 4000
            outline = Rectangle(width=max(0.05, W * spec), height=0.24, stroke_color=col, stroke_width=1.5,
                                fill_color=col, fill_opacity=0.12)
            fill = Rectangle(width=max(0.045, W * (got or spec)), height=0.24, stroke_width=0, fill_color=col,
                             fill_opacity=0.9 if got else 0.45).align_to(outline, LEFT)
            txt = f"~{got:,} of {spec:,} GB/s" if got else f"{spec:,} GB/s (spec)"
            lab = T(f"{name}   {txt}", 17, INK).next_to(outline, UP, buff=0.04, aligned_edge=LEFT)
            bars.add(VGroup(lab, VGroup(outline, fill)))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(rows, DOWN, buff=0.45, aligned_edge=LEFT)
        for gb in bars:
            gb[1].align_to(bars, LEFT)
        hdr = T("per GPU, each way: achievable (filled) of spec (outline)", 16, INK2).next_to(bars, UP, buff=0.1, aligned_edge=LEFT)
        self.play(FadeIn(hdr), run_time=0.5)
        for gb in bars:
            self.play(FadeIn(gb[0]), GrowFromEdge(gb[1], LEFT), run_time=0.7)
            pause(self, "quick")
        pause(self, "quick")
        key = T("C2C: the second-fastest way into the GPU", 21, C2C, weight="BOLD").next_to(bars, DOWN, buff=0.3, aligned_edge=LEFT)
        room = 6.85 - key.get_left()[0]
        if key.width > room:
            key.scale_to_fit_width(room).align_to(bars, LEFT)
        self.play(Indicate(bars[1][1], color=C2C, scale_factor=1.3), run_time=1.0)
        show(self, key, extra=0.5)
        clear(self)


# --------------------------------------------------------------------------
class S2Problem(Scene):
    def construct(self):
        title(self, "The model doesn't fit", "MiMo-V2.6-Pro: ~1T parameters, MXFP4")
        f1 = T("69 MoE layers x 384 experts, top-8 routing", 26).shift(UP * 1.6)
        f2 = T("checkpoint: 566 GB, nearly all of it experts", 26).next_to(f1, DOWN, buff=0.2)
        show(self, VGroup(f1, f2))

        unit = 6.0 / 566
        model = Rectangle(width=566 * unit, height=0.6, fill_color=INK2, fill_opacity=0.8, stroke_width=0)
        hbm = Rectangle(width=384 * unit, height=0.6, fill_color=HBM, fill_opacity=0.9, stroke_width=0)
        VGroup(model, hbm).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.2 + RIGHT * 0.8)
        ml = T("model weights  566 GB", 22).next_to(model, LEFT, buff=0.3)
        hl = T("node HBM  384 GB", 22, HBM).next_to(hbm, LEFT, buff=0.3)
        self.play(GrowFromEdge(model, LEFT), FadeIn(ml), run_time=1.0)
        pause(self, "long")
        self.play(GrowFromEdge(hbm, LEFT), FadeIn(hl), run_time=1.0)
        pause(self, "long")
        gap = DashedLine(hbm.get_right() + UP * 1.2, hbm.get_right() + DOWN * 0.4, color=BAD)
        self.play(Create(gap), run_time=0.6)
        pause(self, "quick")
        over = T("and KV cache + activations need room too", 20, BAD).next_to(hbm, DOWN, buff=0.2).align_to(hbm, LEFT)
        show(self, over)

        a = VGroup(T("Option A: two nodes", 26, weight="BOLD"),
                   T("8 GPUs, but every layer's collectives", 20, INK2),
                   T("cross InfiniBand at 25 GB/s per GPU", 20, INK2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        b = VGroup(T("Option B: offload to Grace RAM", 26, weight="BOLD"),
                   T("480 GB right next to the GPUs,", 20, INK2),
                   T("reachable at 450 GB/s over C2C", 20, INK2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        VGroup(a, b).arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=0.55)
        show(self, a)
        show(self, b)
        self.play(a.animate.set_opacity(0.3), b[0].animate.set_color(DDR), run_time=0.8)
        pause(self, "long")
        clear(self)


# --------------------------------------------------------------------------
def lane(label, color, y, x0=-4.2):
    lab = T(label, 20, color).move_to([x0 - 1.4, y, 0])
    track = Line([x0, y, 0], [5.8, y, 0], color=MUTED, stroke_width=1)
    return VGroup(lab, track)


def seg(x, y, w, color, text=None, h=0.5):
    r = Rectangle(width=w, height=h, fill_color=color, fill_opacity=0.85, stroke_width=0)
    r.move_to([x + w / 2, y, 0])
    if text:
        lab = T(text, 18, BG).move_to(r)
        if lab.width > w * 0.92:
            lab.scale_to_fit_width(w * 0.92)
        return VGroup(r, lab)
    return r


class S3StockVllm(Scene):
    def construct(self):
        title(self, "How stock vLLM offloads", "--cpu-offload-gb")
        pts = [
            T("moves whole layers' weights to CPU memory", 24),
            T("every forward pass streams all of them back over the link", 24),
            T("including experts the tokens never route to", 24, BAD),
        ]
        VGroup(*pts).arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(UP * 1.7)
        show(self, pts[0], shift=RIGHT * 0.2)
        show(self, VGroup(pts[1], pts[2]), shift=RIGHT * 0.2)

        y1, y2 = -0.7, -1.5
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", C2C, y2)
        self.play(FadeIn(l1), FadeIn(l2), run_time=0.6)
        pause(self, "quick")
        x = -4.2
        for _ in range(2):
            a = seg(x, y1, 1.3, HBM, "HBM layer")
            x += 1.3
            b = seg(x, y2, 3.0, C2C, "offloaded layer: all experts")
            x += 3.0
            self.play(GrowFromEdge(a, LEFT), run_time=0.9, rate_func=linear)
            self.play(GrowFromEdge(b, LEFT), run_time=2.0, rate_func=linear)
        pause(self, "long")
        c1 = T("one link busy at a time: HBM idles while C2C works, and back", 22, INK2).to_edge(DOWN, buff=0.75)
        show(self, c1)
        c2 = T("most of the bytes moved are never used", 22, BAD).next_to(c1, DOWN, buff=0.15)
        show(self, c2)
        res = T("~70-80 tok/s at c=1, even with speculative decoding", 26, weight="BOLD").next_to(VGroup(*pts), DOWN, buff=0.35)
        self.play(Write(res), run_time=1.2)
        read(self, res, extra=0.5)
        clear(self)


# --------------------------------------------------------------------------
CELL_W, CELL_GAP, GRID_COLS = 0.16, 0.035, 12


def grid_pos(origin, k):
    r, c = divmod(k, GRID_COLS)
    return origin + RIGHT * (c * (CELL_W + CELL_GAP)) + DOWN * (r * (CELL_W + CELL_GAP))


def cell(color=CELL, opacity=0.9):
    return Rectangle(width=CELL_W, height=CELL_W, stroke_width=0, fill_color=color, fill_opacity=opacity)


def gpu_columns(height):
    return VGroup(*[box(2.75, height, HBM, fill=0.07) for _ in range(4)]).arrange(RIGHT, buff=0.3)


def owned_grids(cols, top_offset):
    """One cell per owned expert, in id order, as on the layout slide."""
    cells = {}
    grid_w = GRID_COLS * (CELL_W + CELL_GAP) - CELL_GAP
    for r, col in enumerate(cols):
        origin = col.get_top() + DOWN * top_offset + LEFT * (grid_w / 2 - CELL_W / 2)
        for k, e in enumerate(STEP["ranks"][r]["owned"]):
            cells[e] = cell().move_to(grid_pos(origin, k))
    return cells


class S4Layout(Scene):
    def construct(self):
        title(self, "Our layout: TP4 attention, EP4 experts", "plus speculative decoding")
        cols = gpu_columns(3.0).shift(DOWN * 0.45)
        heads = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").next_to(c, UP, buff=0.1) for i, c in enumerate(cols)])
        self.play(FadeIn(cols), FadeIn(heads), run_time=0.9)
        pause(self, "look")

        attn = VGroup()
        for c in cols:
            b = box(2.4, 0.5, NVL, fill=0.2).move_to(c.get_top() + DOWN * 0.45)
            attn.add(VGroup(b, T("attention: 1/4 heads", 14).move_to(b)))
        self.play(LaggedStart(*[FadeIn(a) for a in attn], lag_ratio=0.15), run_time=1.0)
        pause(self, "quick")
        cap = T("attention is split by heads across the 4 GPUs (TP4)", 22, INK2).to_edge(DOWN, buff=0.35)
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        cells = owned_grids(cols, 1.05)
        grids = VGroup(*cells.values())
        self.play(LaggedStart(*[FadeIn(c) for c in grids], lag_ratio=0.002), run_time=1.5)
        pause(self, "quick")
        cap = T("each GPU owns 96 of the 384 experts of every layer (EP4)", 22, INK2).to_edge(DOWN, buff=0.35)
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        tok0 = [cells[e] for e in STEP["token0"]]
        self.play(*[c.animate.set_fill(C2C, opacity=1) for c in tok0], run_time=0.8)
        pause(self, "look")
        cap = T("1 token: top-8, so 8 experts per layer", 24, C2C).to_edge(DOWN, buff=0.35)
        show(self, cap, extra=0.25)
        self.play(FadeOut(cap), *[c.animate.set_fill(CELL, opacity=0.9) for c in tok0], run_time=0.6)
        pause(self, "quick")

        sd = VGroup(
            T("but we decode with speculative decoding: a drafter (MTP or DFlash)", 21),
            T("proposes 7 tokens, and the model verifies all 8 in a single step", 21),
        ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.25)
        show(self, sd, extra=0.3)
        self.play(FadeOut(sd), run_time=0.4)

        act = [cells[e] for e in STEP["active"]]
        self.play(LaggedStart(*[c.animate.set_fill(C2C, opacity=1) for c in act], lag_ratio=0.03), run_time=1.6)
        pause(self, "study")
        cap = T("8 tokens per step: ~49 experts per layer, up to 64 (8 tokens x top-8)", 22, C2C).to_edge(DOWN, buff=0.35)
        show(self, cap, extra=0.5)
        self.play(FadeOut(cap), run_time=0.4)
        why = VGroup(
            T("each weight read serves up to 8 tokens, and with more active experts", 21),
            T("the hot/cold split and per-GPU load land closer to their averages", 21),
        ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.25)
        show(self, why, extra=0.4)
        clear(self)


# --------------------------------------------------------------------------
class S4bPrefetch(Scene):
    def construct(self):
        title(self, "First idea: prefetch the next layer",
              "copy layer N+1's cold experts into HBM while layer N runs (two buffers)")
        x0, y_gpu, y_cp = -4.4, 0.2, -0.75
        COPY = 2.1  # copying one layer's cold experts (~0.8 GB per GPU over C2C, ~2 ms)
        cap_y = -2.2
        l_gpu = lane("GPU compute", HBM, y_gpu, x0=x0)
        l_cp = lane("C2C copy", DDR, y_cp, x0=x0)
        self.play(FadeIn(l_gpu), FadeIn(l_cp), run_time=0.6)
        bs = ValueTracker(4096)

        def width(b):
            # Illustrative: a layer's compute grows with the tokens per step;
            # it matches the copy at ~256 tokens.
            lb = np.log2(b)
            if lb <= 8:
                return 0.4 + (COPY - 0.4) * (lb - 3) / 5
            return COPY + 0.95 * (lb - 8) / 4

        def timeline():
            w = width(bs.get_value())
            g = VGroup()
            start, cp_end, prev_end = 0.0, 0.0, 0.0
            copy_start = 0.0
            for i in range(3):
                if i > 0:
                    start = max(prev_end, cp_end)
                    if start > prev_end + 1e-3:
                        gap = Rectangle(width=start - prev_end, height=0.5, stroke_color=BAD,
                                        stroke_width=2, fill_color=BAD, fill_opacity=0.12)
                        gap.move_to([x0 + (prev_end + start) / 2, y_gpu, 0])
                        g.add(gap)
                        if start - prev_end > 0.7:
                            g.add(T("idle", 14, BAD).move_to(gap))
                g.add(seg(x0 + start, y_gpu, w - 0.05, HBM, f"layer {i + 1}"))
                # The copy for the next layer starts once the previous copy is
                # done and its buffer's last reader has started.
                copy_start = max(cp_end, start)
                cp_end = copy_start + COPY
                g.add(seg(x0 + copy_start, y_cp, COPY - 0.05, DDR, f"copy layer {i + 2}"))
                prev_end = start + w
            return g

        tl = always_redraw(timeline)
        tag = always_redraw(lambda: T(f"{int(round(bs.get_value()))} tokens per step", 22, C2C, weight="BOLD")
                            .move_to([0, 1.25, 0]))
        self.play(FadeIn(tl), FadeIn(tag), run_time=0.8)
        pause(self, "look")
        cap = T("prefill: lots of compute per layer, so each copy hides behind it - offloading is free",
                21, GOOD).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        self.play(bs.animate.set_value(8), run_time=3.0)
        pause(self, "study")
        cap = VGroup(
            T("decode, 8 tokens per step: a layer takes ~0.25 ms, its copy ~2 ms", 21, BAD),
            T("the GPU spends most of the step waiting for copies", 21, BAD),
        ).arrange(DOWN, buff=0.08).move_to([0, cap_y - 0.1, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        self.play(bs.animate.set_value(256), run_time=2.5)
        pause(self, "look")
        cap = T("the copies hide once a step has roughly 256 tokens or more", 21).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)
        cap = VGroup(
            T("so prefill prefetches, and offloading costs it nothing", 22, GOOD, weight="BOLD"),
            T("decode needs something smarter", 22, weight="BOLD"),
        ).arrange(DOWN, buff=0.1).move_to([0, cap_y - 0.1, 0])
        show(self, cap, extra=0.4)
        clear(self)


# --------------------------------------------------------------------------
class S5Overlap(Scene):
    def construct(self):
        title(self, "Idea 1: read both memories at once", "decode is bandwidth-bound")
        g, h = grace(), hopper()
        h.next_to(g, RIGHT, buff=0.9)
        link = Line(g.get_right(), h.get_left(), color=C2C, stroke_width=10)
        chip = VGroup(g, h, link).scale(0.85).move_to(UP * 1.6)
        chip.set_x(0)
        self.play(FadeIn(chip), run_time=0.7)
        pause(self, "quick")
        hot = T("hot experts live in HBM: Marlin reads them at ~2.2 TB/s", 22, HBM).next_to(chip, DOWN, buff=0.35)
        show(self, hot)
        cold = T("cold experts live in Grace RAM: read over C2C at ~420 GB/s", 22, DDR).next_to(hot, DOWN, buff=0.12)
        show(self, cold)

        # Bar length = time. HBM at ~2.2 TB/s vs C2C at ~0.42 TB/s: one cold
        # expert takes about as long as 5 hot ones; drawn as exactly 5.
        x0 = -4.2
        wh = 1.0
        wc = 5 * wh
        y1, y2, y3 = -0.95, -1.65, -2.35
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", DDR, y2)
        note = T("bar length = time", 16, MUTED).next_to(l1, UP, buff=0.3).align_to(l1, RIGHT)
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(note), run_time=0.6)
        pause(self, "quick")

        def hot_blocks(n, y=y1):
            return VGroup(*[
                Rectangle(width=wh - 0.04, height=0.5, fill_color=HBM, fill_opacity=0.85, stroke_width=0)
                .move_to([x0 + (i + 0.5) * wh, y, 0])
                for i in range(n)
            ])

        def hot_label(blocks, n):
            return T(f"{n} hot experts", 16, HBM).next_to(blocks, UP, buff=0.05).align_to(blocks, LEFT)

        def cold_block(x):
            return seg(x, y2, wc, DDR, "1 cold expert")

        hb = hot_blocks(5)
        hl = hot_label(hb, 5)
        cb = cold_block(x0 + 5 * wh)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in hb], lag_ratio=0.9), run_time=1.2, rate_func=linear)
        self.play(FadeIn(hl), GrowFromEdge(cb, LEFT), run_time=1.2, rate_func=linear)
        pause(self, "look")
        cap = T("one after the other: the layer takes hot time + cold time", 22, BAD).to_edge(DOWN, buff=0.3)
        show(self, cap)
        self.play(cb.animate.shift(LEFT * 5 * wh), FadeOut(cap), run_time=1.2)
        pause(self, "look")
        cap = T("at the same time: the layer takes only as long as the longer of the two", 22, GOOD).to_edge(DOWN, buff=0.3)
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        def case(n, text, color, mark):
            nb = hot_blocks(n)
            self.play(Transform(hb, nb), Transform(hl, hot_label(nb, n)), run_time=1.0)
            pause(self, "quick")
            self.play(FadeIn(mark), run_time=0.6)
            pause(self, "quick")
            c = T(text, 22, color).to_edge(DOWN, buff=0.3)
            show(self, c)
            self.play(FadeOut(VGroup(mark, c)), run_time=0.4)

        idle = Rectangle(width=wc - 3 * wh, height=0.5, stroke_color=BAD, stroke_width=2, fill_opacity=0)
        idle.move_to([x0 + (3 * wh + wc) / 2, y1, 0])
        case(3, "3 hot : 1 cold - the cold read runs longer and the GPU waits on it", BAD,
             VGroup(idle, T("waiting", 15, BAD).move_to(idle)))
        hidden = SurroundingRectangle(cb, color=GOOD, buff=0.05)
        case(6, "6 hot : 1 cold - the cold read is fully overlapped by the hot ones", GOOD,
             VGroup(hidden, T("fully overlapped", 15, GOOD).next_to(hidden, DOWN, buff=0.05).align_to(hidden, LEFT)))
        nb = hot_blocks(5)
        self.play(Transform(hb, nb), Transform(hl, hot_label(nb, 5)), run_time=1.0)
        end_line = DashedLine([x0 + wc, y1 + 0.45, 0], [x0 + wc, y2 - 0.35, 0], color=C2C)
        self.play(Create(end_line), run_time=0.5)
        pause(self, "look")
        c = VGroup(
            T("5 hot : 1 cold - both links stay busy and finish together", 22, C2C, weight="BOLD"),
            T("one cold expert takes about as long as 5 hot ones (~2.2 vs ~0.42 TB/s)", 20, INK2),
        ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.2)
        show(self, c, extra=0.6)
        self.play(FadeOut(c), run_time=0.4)

        # Faster than having everything in HBM.
        l3 = lane("all in HBM", INK2, y3)
        ab = hot_blocks(6, y3).set_fill(INK2, opacity=0.6)
        self.play(FadeIn(l3), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in ab], lag_ratio=0.9), run_time=1.4, rate_func=linear)
        saved = Rectangle(width=6 * wh - wc, height=0.5, stroke_color=GOOD, stroke_width=2, fill_color=GOOD, fill_opacity=0.15)
        saved.move_to([x0 + (wc + 6 * wh) / 2, y3, 0])
        saved_lab = T("saved", 15, GOOD).move_to(saved)
        self.play(FadeIn(saved), FadeIn(saved_lab), run_time=0.6)
        pause(self, "study")
        c1 = T("the same 6 experts, read only from HBM, finish later", 22, INK).to_edge(DOWN, buff=0.35)
        show(self, c1)
        self.play(FadeOut(c1), run_time=0.3)
        c = VGroup(
            T("Offloading is faster than VRAM!", 32, GOOD, weight="BOLD"),
            T("C2C adds its bandwidth on top of HBM's: ~2.6 TB/s instead of 2.2", 20, INK2),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(c[0], scale=1.2), Indicate(saved, color=GOOD, scale_factor=1.15), run_time=1.0)
        self.play(FadeIn(c[1]), run_time=0.5)
        read(self, c, extra=0.6)
        par = VGroup(hb, hl, cb, end_line, l3, ab, saved, saved_lab)
        free = c

        self.play(*[FadeOut(m) for m in (par, l1, l2, note, hot, cold, free)], run_time=0.7)
        better = VGroup(
            T("Even better: one kernel reads both tiers", 30, weight="BOLD"),
            T("instead of two kernels on two streams", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to(UP * 0.35)
        show(self, better)
        sms = VGroup(*[Rectangle(width=0.28, height=0.28, stroke_width=0, fill_color=HBM, fill_opacity=0.85) for _ in range(132)])
        sms.arrange_in_grid(6, 22, buff=0.05).next_to(better, DOWN, buff=0.35)
        lbl = T("132 SMs, one launch", 18, INK2).next_to(sms, DOWN, buff=0.12)
        self.play(LaggedStart(*[FadeIn(s) for s in sms], lag_ratio=0.005), FadeIn(lbl), run_time=1.3)
        pause(self, "look")
        self.play(VGroup(*sms[:20]).animate.set_fill(DDR), run_time=0.9)
        pause(self, "look")
        l1 = T("~20 SMs stream cold experts from Grace: enough to saturate C2C", 20, DDR).next_to(lbl, DOWN, buff=0.2)
        l2 = T("the other ~112 stream hot experts from HBM, at the same time", 20, HBM).next_to(l1, DOWN, buff=0.1)
        show(self, VGroup(l1, l2), extra=0.5)
        self.play(FadeOut(VGroup(l1, l2)), run_time=0.4)
        gain = T("measured per layer: 6-32% faster than two kernels on two streams", 23, GOOD, weight="BOLD").next_to(lbl, DOWN, buff=0.3)
        show(self, gain, extra=0.6)
        clear(self)


# --------------------------------------------------------------------------
class S6Problem(Scene):
    def construct(self):
        title(self, "The problem: we have to offload much more than ~18%",
              "of the experts for the model to run at all")
        cold_frac = 1 - DATA["hot_fraction_of_experts"]
        bar = VGroup(
            Rectangle(width=10 * (1 - cold_frac), height=0.6, fill_color=HBM, fill_opacity=0.9, stroke_width=0),
            Rectangle(width=10 * cold_frac, height=0.6, fill_color=DDR, fill_opacity=0.9, stroke_width=0),
        ).arrange(RIGHT, buff=0).shift(UP * 1.2)
        self.play(GrowFromEdge(bar, LEFT), run_time=1.2)
        pause(self, "quick")
        labs = VGroup(
            T(f"{1-cold_frac:.0%} of experts fit in HBM", 20).next_to(bar[0], DOWN, buff=0.12),
            T(f"{cold_frac:.0%} must live in Grace RAM", 20).next_to(bar[1], DOWN, buff=0.12),
        )
        show(self, labs)
        q = T("if every expert were used equally often...", 26).shift(DOWN * 0.1)
        show(self, q)

        y1, y2 = -1.3, -2.1
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", DDR, y2)
        slow = (cold_frac / 0.42) / ((1 - cold_frac) / 2.2)
        hotb = seg(-4.2, y1, 1.2, HBM, "hot")
        coldb = seg(-4.2, y2, 1.2 * slow, DDR, f"cold: ~{slow:.0f}x longer than hot")
        self.play(FadeIn(l1), FadeIn(l2), run_time=0.5)
        self.play(GrowFromEdge(hotb, LEFT), GrowFromEdge(coldb, LEFT), run_time=2.5, rate_func=linear)
        pause(self, "long")
        ww = 1.2 * slow - 1.2
        wait = Rectangle(width=ww, height=0.5, stroke_color=BAD, stroke_width=2, fill_opacity=0).move_to([-4.2 + 1.2 + ww / 2, y1, 0])
        self.play(Create(wait), FadeIn(T("GPU waits", 16, BAD).move_to(wait)), run_time=0.8)
        pause(self, "look")
        msg = T(f"~{cold_frac:.0%} of the routes on the slow link: cold sets the step time", 23, BAD).to_edge(DOWN, buff=0.3)
        show(self, msg, extra=0.5)
        clear(self)


# --------------------------------------------------------------------------
class S7Frequency(Scene):
    def construct(self):
        title(self, "Idea 2: offload the experts nobody uses", "routing is far from uniform")
        share = np.array(DATA["sorted_share_layer"])
        n = len(share)
        W, H = 10.5, 3.3
        x0, y0 = -W / 2, -2.45
        unit = H / share.max()
        bars = VGroup()
        for i, s in enumerate(share):
            r = Rectangle(width=W / n, height=max(0.005, s * unit), stroke_width=0, fill_color=INK2, fill_opacity=0.9)
            r.move_to([x0 + (i + 0.5) * W / n, y0, 0], aligned_edge=DOWN)
            bars.add(r)
        axis = Line([x0, y0, 0], [x0 + W, y0, 0], color=MUTED)
        xl = T(f"the 384 experts of layer {DATA['layer_shown']}, sorted by how often they are routed to", 18, INK2).next_to(axis, DOWN, buff=0.12)
        self.play(Create(axis), FadeIn(xl), run_time=0.8)
        pause(self, "quick")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.004), run_time=3.0)
        pause(self, "study")
        fair = DashedLine([x0, y0 + unit / n, 0], [x0 + W, y0 + unit / n, 0], color=C2C)
        fl = T("fair share", 16, C2C).next_to(fair, UP, buff=0.05).align_to(fair, RIGHT)
        self.play(Create(fair), FadeIn(fl), run_time=0.8)
        pause(self, "look")
        c1 = T("a few experts take most of the traffic; the long tail barely runs", 22).move_to([0, 1.55, 0])
        show(self, c1, extra=0.25)
        self.play(FadeOut(c1), run_time=0.4)
        src = VGroup(
            T("profile on a calibration set that looks like deployment", 22, weight="BOLD"),
            T("for us: agentic coding sessions (Claude Code) + autoresearch ML tasks", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.55, 0])
        show(self, src[0])
        show(self, src[1], extra=0.25)
        self.play(FadeOut(src), run_time=0.4)

        half = n // 2
        self.play(*[b.animate.set_fill(HBM) for b in bars[:half]], *[b.animate.set_fill(DDR) for b in bars[half:]], run_time=1.3)
        pause(self, "long")
        lh = T(f"least-used half -> Grace RAM: only {DATA['cold_share_least_used_half']:.1%} of all routes", 23, DDR).move_to([0, 1.55, 0])
        show(self, lh, extra=0.4)
        self.play(FadeOut(lh), run_time=0.4)
        fin = VGroup(
            T(f"production placement: {DATA['cold_share_profile']:.1%} of routes go to cold experts", 23, GOOD, weight="BOLD"),
            T("within the ~5:1 the overlap can hide", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.55, 0])
        show(self, fin[0])
        show(self, fin[1], extra=0.5)

        # The skew depends on the model: GLM-5.3 next to MiMo, same scale
        # (multiples of each model's fair share).
        GLM = json.loads((HERE / "glm.json").read_text())
        self.play(FadeOut(fin), FadeOut(xl), FadeOut(fl), run_time=0.4)
        k, base_y = 0.55, -2.1
        mimo = VGroup(bars, axis, fair)
        self.play(mimo.animate.scale(k, about_point=axis.get_left()).shift(
            [-6.3 - axis.get_left()[0], base_y - axis.get_left()[1], 0]), run_time=1.0)
        gshare = np.array(GLM["sorted_share_layer"])
        gn = len(gshare)
        gunit = unit * gn / n  # same height per multiple of fair share
        gbars = VGroup()
        for i, sh in enumerate(gshare):
            r = Rectangle(width=W / gn, height=max(0.005, sh * gunit), stroke_width=0,
                          fill_color=HBM if i < gn // 2 else DDR, fill_opacity=0.9)
            r.move_to([x0 + (i + 0.5) * W / gn, y0, 0], aligned_edge=DOWN)
            gbars.add(r)
        gaxis = Line([x0, y0, 0], [x0 + W, y0, 0], color=MUTED)
        gfair = DashedLine([x0, y0 + gunit / gn, 0], [x0 + W, y0 + gunit / gn, 0], color=C2C)
        glm = VGroup(gbars, gaxis, gfair)
        glm.scale(k, about_point=gaxis.get_left()).shift([0.5 - gaxis.get_left()[0], base_y - gaxis.get_left()[1], 0])
        names = VGroup(
            T("MiMo-V2.6-Pro (384 experts)", 20, weight="BOLD").move_to([mimo.get_center()[0], 0.55, 0]),
            T("GLM-5.3 (256 experts)", 20, weight="BOLD").move_to([glm.get_center()[0], 0.55, 0]),
        )
        self.play(FadeIn(names[0]), run_time=0.4)
        self.play(Create(gaxis), FadeIn(names[1]), run_time=0.5)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in gbars], lag_ratio=0.006), Create(gfair), run_time=2.0)
        pause(self, "look")
        halves = VGroup(
            T(f"least-used half: {DATA['cold_share_least_used_half']:.1%} of routes", 18, DDR)
            .move_to([mimo.get_center()[0], base_y - 0.35, 0]),
            T(f"least-used half: {GLM['cold_share_least_used_half']:.1%} of routes", 18, DDR)
            .move_to([glm.get_center()[0], base_y - 0.35, 0]),
        )
        show(self, halves)
        cap = VGroup(
            T("GLM-5.3 routes more evenly than MiMo", 22, weight="BOLD"),
            T("how much there is to gain from placement depends on the model", 20, INK2),
        ).arrange(DOWN, buff=0.08).move_to([0, 1.55, 0])
        show(self, cap, extra=0.5)
        clear(self)


# --------------------------------------------------------------------------
# One real decode layer (MiMo layer 26, step 20) on the four GPUs, in us before
# the layer's reduce-scatter completes; read off figs/layer-ranks.png.
LAYER26 = [
    # (hot start, hot end, cold start, cold end, wait start)
    (-282, -158, -278, -200, -148),
    (-281, -152, -279, -17, -7),
    (-281, -197, -279, -17, -7),
    (-280, -173, -280, -268, -165),
]


class S8Imbalance(Scene):
    def construct(self):
        title(self, "Even then: the GPUs don't finish together", "one real decode layer on the four GPUs")
        x_l, x_r = -4.9, 5.9
        t0, t1 = -330.0, 0.0
        u = (x_r - x_l) / (t1 - t0)

        def X(t):
            return x_l + (t - t0) * u

        ys = [1.35, 0.35, -0.65, -1.65]
        hh = 0.3
        names = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").move_to([x_l - 0.85, y, 0]) for i, y in enumerate(ys)])
        axis = Line([x_l, -2.35, 0], [x_r, -2.35, 0], color=MUTED)
        ticks = VGroup(*[T(f"{t}", 15, INK2).move_to([X(t), -2.6, 0]) for t in (-300, -200, -100, 0)])
        xlab = T("microseconds before the layer's collective completes (together on all GPUs)", 17, INK2).move_to([0.5, -2.95, 0])
        self.play(FadeIn(names), Create(axis), FadeIn(ticks), FadeIn(xlab), run_time=0.9)
        legend = VGroup(*[
            VGroup(Rectangle(width=0.35, height=0.2, fill_color=c, fill_opacity=1, stroke_width=0), T(t, 16, INK2)).arrange(RIGHT, buff=0.12)
            for c, t in ((OTHER, "rest of the layer"), (HBM, "hot experts (HBM)"), (DDR, "cold experts (Grace)"), (WAIT, "waiting in the collective"))
        ]).arrange(RIGHT, buff=0.5).move_to([0.3, 2.35, 0])
        self.play(FadeIn(legend), run_time=0.6)
        pause(self, "look")

        now = ValueTracker(t0)

        def bar(start, end, y, color):
            def make():
                t = now.get_value()
                w = (min(t, end) - start) * u
                r = Rectangle(width=max(w, 1e-3), height=hh, stroke_width=0, fill_color=color,
                              fill_opacity=0.95 if t > start else 0)
                return r.move_to([X(start) + max(w, 1e-3) / 2, y, 0])
            return always_redraw(make)

        segs = VGroup()
        for i, (hs, he, cs, ce, ws) in enumerate(LAYER26):
            y = ys[i]
            segs.add(bar(t0, min(hs, cs), y, OTHER))
            segs.add(bar(hs, he, y + hh / 2 + 0.01, HBM))
            segs.add(bar(cs, ce, y - hh / 2 - 0.01, DDR))
            segs.add(bar(ws, 0, y, WAIT))
        cursor = always_redraw(lambda: Line([X(now.get_value()), 1.75, 0], [X(now.get_value()), -2.35, 0], color=INK, stroke_width=1.5))
        self.add(segs, cursor)
        self.play(now.animate.set_value(t1), run_time=6, rate_func=linear)
        self.remove(cursor)
        pause(self, "study")

        cap_y = -3.45
        c1 = T("each GPU reads its hot and cold experts at the same time", 21).move_to([0, cap_y, 0])
        show(self, c1)
        self.play(FadeOut(c1), run_time=0.4)
        hl = VGroup(SurroundingRectangle(VGroup(segs[6], segs[10]), color=DDR, buff=0.06))
        self.play(Create(hl), run_time=0.8)
        pause(self, "quick")
        c2 = T("GPUs 1 and 2 drew lots of cold experts: cold sets their pace", 21, DDR).move_to([0, cap_y, 0])
        show(self, c2)
        self.play(FadeOut(c2), FadeOut(hl), run_time=0.4)
        hl = VGroup(SurroundingRectangle(segs[3], color=WAIT, buff=0.06), SurroundingRectangle(segs[15], color=WAIT, buff=0.06))
        self.play(Create(hl), run_time=0.8)
        pause(self, "quick")
        c3 = T("GPUs 0 and 3 finish early and sit in the collective, ~150 us doing nothing", 21, WAIT).move_to([0, cap_y, 0])
        show(self, c3, extra=0.25)
        self.play(FadeOut(c3), FadeOut(hl), run_time=0.4)
        c4 = T("which GPU is last changes every layer: routing luck, not a slow GPU", 21).move_to([0, cap_y, 0])
        show(self, c4, extra=0.5)
        clear(self)


class S9Replicas(Scene):
    def construct(self):
        title(self, "Idea 3: replicas for the cold experts", "back to the per-GPU layout")
        cols = gpu_columns(4.3).shift(DOWN * 0.75)
        heads = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").next_to(c, UP, buff=0.08) for i, c in enumerate(cols)])
        cells = owned_grids(cols, 0.5)
        self.play(FadeIn(cols), FadeIn(heads), FadeIn(VGroup(*cells.values())), run_time=0.8)
        pause(self, "quick")
        cap_y = 2.35

        # Each GPU's experts split into an HBM panel and a Grace panel.
        grid_w = GRID_COLS * (CELL_W + CELL_GAP) - CELL_GAP
        panels, plabels, origins = VGroup(), VGroup(), []
        for r, col in enumerate(cols):
            hp = box(2.55, 1.35, HBM, fill=0.1).move_to(col.get_top() + DOWN * 1.0)
            gp = box(2.55, 1.75, DDR, fill=0.1).next_to(hp, DOWN, buff=0.35)
            panels.add(hp, gp)
            plabels.add(T("HBM", 14, HBM).next_to(hp, UP, buff=0.03).align_to(hp, LEFT),
                        T("Grace RAM", 14, DDR).next_to(gp, UP, buff=0.03).align_to(gp, LEFT))
            origins.append((hp.get_top() + DOWN * 0.22 + LEFT * (grid_w / 2 - CELL_W / 2),
                            gp.get_top() + DOWN * 0.22 + LEFT * (grid_w / 2 - CELL_W / 2)))
        moves = []
        for r in range(4):
            info = STEP["ranks"][r]
            for k, e in enumerate(info["hot"]):
                moves.append(cells[e].animate.move_to(grid_pos(origins[r][0], k)).set_fill(HBM, opacity=0.55))
            for k, e in enumerate(info["cold"]):
                moves.append(cells[e].animate.move_to(grid_pos(origins[r][1], k)).set_fill(DDR, opacity=0.55))
        self.play(FadeIn(panels), FadeIn(plabels), run_time=0.6)
        self.play(*moves, run_time=1.6)
        pause(self, "look")
        cap = T("after placement: hot experts in HBM, cold ones in that GPU's Grace RAM", 21, INK2).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        # The real verify step.
        cold_act = STEP["cold_active"]
        hot_act = [e for e in STEP["active"] if e not in set(cold_act)]
        self.play(*[cells[e].animate.set_fill(INK, opacity=1) for e in hot_act], run_time=0.8)
        self.play(*[cells[e].animate.set_fill(DDR, opacity=1).set_stroke(INK, 1.5) for e in cold_act], run_time=0.8)
        pause(self, "look")
        cap = T(f"one real verify step (layer {STEP['layer']}): {len(STEP['active'])} experts active, {len(cold_act)} of them cold",
                21).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        # Pull this step's cold experts out into one stack per GPU.
        BW, BH, GAP = 1.9, 0.4, 0.05

        def slot(r, k):
            base = cols[r].get_bottom()[1] + 0.2
            return [cols[r].get_center()[0], base + BH / 2 + k * (BH + GAP), 0]

        def block(e, copy=False):
            r = Rectangle(width=BW, height=BH, stroke_color=DDR, stroke_width=2,
                          fill_color=DDR, fill_opacity=0.18 if copy else 0.9)
            lab = T(f"#{e} copy" if copy else f"#{e}", 15, INK if copy else BG, weight="BOLD")
            return VGroup(r, lab.move_to(r))

        owner = {int(e): v for e, v in STEP["owner"].items()}
        stacks = {r: sorted(e for e in cold_act if owner[e] == r) for r in range(4)}
        blocks = {}
        for r in range(4):
            for k, e in enumerate(stacks[r]):
                blocks[e] = block(e).move_to(slot(r, k))
        rest = [cells[e] for e in cells if e not in set(cold_act)]
        self.play(FadeOut(panels), FadeOut(plabels), *[FadeOut(c) for c in rest],
                  *[ReplacementTransform(cells[e], blocks[e]) for e in cold_act], run_time=1.4)
        pause(self, "look")
        before = STEP["no_rep"]
        worst = int(np.argmax(before))

        def wait_line(n, color, text):
            y = slot(0, n)[1] - BH / 2 - GAP / 2 + 0.02
            line = DashedLine([cols.get_left()[0], y, 0], [cols.get_right()[0], y, 0], color=color)
            lab = T(text, 18, color, weight="BOLD").next_to(line, UP, buff=0.06).align_to(line, LEFT).shift(RIGHT * 0.15)
            return VGroup(line, lab)

        wl = wait_line(max(before), BAD, f"the layer waits for {max(before)} cold reads")
        self.play(Create(wl[0]), FadeIn(wl[1]), run_time=0.7)
        pause(self, "look")
        cap = T(f"GPU {worst} drew {max(before)} of the {len(cold_act)} cold experts: the other GPUs wait for it",
                21, BAD).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        # Copies of GPU 3's experts that other GPUs hold.
        cap = T("spare Grace RAM holds copies of other GPUs' cold experts", 21).move_to([0, cap_y, 0])
        show(self, cap)
        assigned = {int(e): r for e, r in STEP["assigned"].items()}
        moved = sorted((e, owner[e], r) for e, r in assigned.items() if owner[e] != r)
        extra = {r: 0 for r in range(4)}
        ghosts, arrows = {}, VGroup()
        for e, o, r in moved:
            g = block(e, copy=True).move_to(slot(r, len(stacks[r]) + extra[r]))
            extra[r] += 1
            ghosts[e] = g
            src, dst = (blocks[e].get_left(), g.get_right()) if r < o else (blocks[e].get_right(), g.get_left())
            arrows.add(Arrow(src, dst, buff=0.05, stroke_width=3, color=C2C,
                             max_tip_length_to_length_ratio=0.06))
        self.play(LaggedStart(*[FadeIn(g) for g in ghosts.values()], lag_ratio=0.3), run_time=1.0)
        pause(self, "quick")
        self.play(FadeOut(cap), LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.3), run_time=1.2)
        pause(self, "look")
        cap = T(f"so GPU {worst} hands {len(moved)} of its cold reads to GPUs that hold a copy", 21).move_to([0, cap_y, 0])
        show(self, cap)

        # The handoff: originals leave, copies run, GPU 3's stack settles.
        keep = {r: [e for e in stacks[r] if assigned[e] == r] for r in range(4)}
        anims = [FadeOut(arrows)]
        for e, o, r in moved:
            anims.append(FadeOut(blocks[e]))
            anims.append(ghosts[e][0].animate.set_fill(DDR, opacity=0.55))
        for r in range(4):
            for k, e in enumerate(keep[r]):
                anims.append(blocks[e].animate.move_to(slot(r, k)))
        after = STEP["with_rep"]
        wl2 = wait_line(max(after), GOOD, f"now it waits for {max(after)}")
        self.play(*anims, FadeOut(cap), run_time=1.3)
        self.play(ReplacementTransform(wl, wl2), run_time=0.9)
        pause(self, "study")
        cap = T("every GPU sees the same routing, so all four pick the same copies without talking",
                21).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)
        stats = T(
            f"over all steps: busiest GPU {DATA['max_cold_rank_no_replicas_mean']:.2f} -> "
            f"{DATA['max_cold_rank_replicas_mean']:.2f} cold reads (perfect balance {DATA['ideal_max_cold_rank_mean']:.2f})",
            20, INK2,
        ).move_to([0, cap_y, 0])
        show(self, stats, extra=0.5)
        clear(self)


# --------------------------------------------------------------------------
class S10Result(Scene):
    def construct(self):
        title(self, "Putting it together", "MiMo-V2.6-Pro, 4x GH200, one user (c=1)")
        items = [
            T("1. hot and cold experts read at the same time, in one kernel", 24),
            T("2. placement from a deployment-like calibration profile", 24),
            T("3. replicas balance the cold work across GPUs", 24),
            T("prefill: copy layer N+1's cold experts into HBM during layer N (double buffered)", 20, INK2),
        ]
        VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(UP * 0.9)
        for it in items:
            show(self, it, extra=-0.4, shift=RIGHT * 0.2)
        unit = 5.0 / 210
        a = Rectangle(width=75 * unit, height=0.5, fill_color=MUTED, fill_opacity=0.9, stroke_width=0)
        b = Rectangle(width=210 * unit, height=0.5, fill_color=GOOD, fill_opacity=0.9, stroke_width=0)
        VGroup(a, b).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 2.0 + RIGHT * 1.6)
        al = T("stock vLLM offload + SD   ~70-80 tok/s", 20).next_to(a, LEFT, buff=0.3)
        bl = T("tiered MoE   ~210 tok/s", 20, GOOD, weight="BOLD").next_to(b, LEFT, buff=0.3)
        self.play(GrowFromEdge(a, LEFT), FadeIn(al), run_time=1.0)
        pause(self, "long")
        self.play(GrowFromEdge(b, LEFT), FadeIn(bl), run_time=1.8)
        self.wait(3)
