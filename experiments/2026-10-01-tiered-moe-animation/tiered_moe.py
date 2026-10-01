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


def read(scene, *mobs, extra=0.0):
    """Hold long enough to read what was just shown."""
    n = sum(chars(m) for m in mobs)
    scene.wait(min(7.5, max(2.2, 1.4 + 0.06 * n)) + extra)


def show(scene, mob, extra=0.0, shift=UP * 0.15):
    scene.play(FadeIn(mob, shift=shift), run_time=0.8)
    read(scene, mob, extra=extra)


def title(scene, text, sub=None):
    t = T(text, 40, weight="BOLD").to_edge(UP, buff=0.35)
    group = VGroup(t)
    if sub:
        group.add(T(sub, 24, INK2).next_to(t, DOWN, buff=0.15))
    scene.play(FadeIn(group, shift=DOWN * 0.2), run_time=0.9)
    read(scene, group, extra=-0.5)
    return group


def box(w, h, color, fill=0.16):
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color,
                            stroke_width=3, fill_color=color, fill_opacity=fill)


def clear(scene, run_time=0.7):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)
    scene.wait(0.3)


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
        self.wait(2.5)
        link = Line(g.get_right(), h.get_left(), color=C2C, stroke_width=12)
        lab = T("NVLink-C2C", 24, C2C, weight="BOLD").next_to(link, UP, buff=0.15)
        self.play(Create(link), FadeIn(lab), run_time=1.0)
        self.wait(1.5)
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
        self.wait(2.5)

        arcs = VGroup()
        for i in range(4):
            for j in range(i + 1, 4):
                d = j - i
                a = cols[i].gpu.get_bottom() + RIGHT * (0.18 * (j - 2.5))
                b = cols[j].gpu.get_bottom() + LEFT * (0.18 * (1.5 - i))
                arcs.add(ArcBetweenPoints(a, b, angle=PI * (0.35 + 0.12 * d), color=NVL, stroke_width=3))
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=1.8)
        self.wait(1.2)
        nv = T("NVLink 4: every GPU pair, 150 GB/s each way", 24, NVL).to_edge(DOWN, buff=0.35)
        show(self, nv)
        self.play(FadeOut(arcs), FadeOut(nv), run_time=0.7)
        more = VGroup(
            T("Grace CPUs link to each other at 100 GB/s each way", 22, INK2),
            T("node to node: 4x InfiniBand NDR200, 25 GB/s each", 22, INK2),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.3)
        show(self, more)
        self.play(FadeOut(more), run_time=0.5)

        node = VGroup(cols, names)
        self.play(node.animate.scale(0.62).to_edge(LEFT, buff=0.35).shift(DOWN * 0.2), run_time=1.2)
        rows = VGroup(
            T("Per node", 28, weight="BOLD"),
            T("384 GB HBM3  (4 x 96)", 24, HBM),
            T("480 GB LPDDR5X  (4 x 120)", 24, DDR),
            T("~990 TFLOPS BF16 per GPU", 24, INK2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.7).shift(UP * 1.6)
        self.play(FadeIn(rows[0]), run_time=0.5)
        for r in rows[1:]:
            self.play(FadeIn(r, shift=LEFT * 0.2), run_time=0.6)
            read(self, r, extra=-0.8)

        ladder = [("HBM -> GPU", "4,000", 4000, HBM), ("Grace RAM -> GPU (C2C)", "450", 450, C2C),
                  ("GPU <-> GPU (NVLink)", "150", 150, NVL), ("node <-> node (IB)", "25", 25, BAD)]
        bars = VGroup()
        for name, txt, gbs, col in ladder:
            b = Rectangle(width=max(0.05, 4.3 * gbs / 4000), height=0.26, fill_color=col, fill_opacity=0.9, stroke_width=0)
            lab = T(f"{name}   {txt} GB/s", 18, INK).next_to(b, UP, buff=0.05, aligned_edge=LEFT)
            bars.add(VGroup(lab, b))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(rows, DOWN, buff=0.45, aligned_edge=LEFT)
        for gb in bars:
            gb[1].align_to(bars, LEFT)
        hdr = T("bandwidth into one GPU, each way", 18, INK2).next_to(bars, UP, buff=0.12, aligned_edge=LEFT)
        self.play(FadeIn(hdr), run_time=0.5)
        for gb in bars:
            self.play(FadeIn(gb[0]), GrowFromEdge(gb[1], LEFT), run_time=0.8)
            self.wait(1.3)
        self.wait(1.0)
        key = T("C2C: the second-fastest way into the GPU", 21, C2C, weight="BOLD").next_to(bars, DOWN, buff=0.3, aligned_edge=LEFT)
        room = 6.85 - key.get_left()[0]
        if key.width > room:
            key.scale_to_fit_width(room).align_to(bars, LEFT)
        self.play(Indicate(bars[1][1], color=C2C, scale_factor=1.3), run_time=1.0)
        show(self, key, extra=1.0)
        clear(self)


# --------------------------------------------------------------------------
class S2Problem(Scene):
    def construct(self):
        title(self, "The model doesn't fit", "MiMo-V2.6-Pro: ~1T parameters, MXFP4")
        f1 = T("69 MoE layers x 384 experts, top-8 routing", 26).shift(UP * 1.6)
        show(self, f1)
        f2 = T("checkpoint: 566 GB, nearly all of it experts", 26).next_to(f1, DOWN, buff=0.2)
        show(self, f2)

        unit = 6.0 / 566
        model = Rectangle(width=566 * unit, height=0.6, fill_color=INK2, fill_opacity=0.8, stroke_width=0)
        hbm = Rectangle(width=384 * unit, height=0.6, fill_color=HBM, fill_opacity=0.9, stroke_width=0)
        VGroup(model, hbm).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.2 + RIGHT * 0.8)
        ml = T("model weights  566 GB", 22).next_to(model, LEFT, buff=0.3)
        hl = T("node HBM  384 GB", 22, HBM).next_to(hbm, LEFT, buff=0.3)
        self.play(GrowFromEdge(model, LEFT), FadeIn(ml), run_time=1.0)
        self.wait(1.8)
        self.play(GrowFromEdge(hbm, LEFT), FadeIn(hl), run_time=1.0)
        self.wait(2.0)
        gap = DashedLine(hbm.get_right() + UP * 1.2, hbm.get_right() + DOWN * 0.4, color=BAD)
        self.play(Create(gap), run_time=0.6)
        self.wait(1.0)
        over = T("and KV cache + activations need room too", 20, BAD).next_to(hbm, RIGHT, buff=0.3)
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
        self.wait(2.5)
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
        for p in pts:
            show(self, p, shift=RIGHT * 0.2)

        y1, y2 = -0.7, -1.5
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", C2C, y2)
        self.play(FadeIn(l1), FadeIn(l2), run_time=0.6)
        self.wait(0.8)
        x = -4.2
        for _ in range(2):
            a = seg(x, y1, 1.3, HBM, "HBM layer")
            x += 1.3
            b = seg(x, y2, 3.0, C2C, "offloaded layer: all experts")
            x += 3.0
            self.play(GrowFromEdge(a, LEFT), run_time=0.9, rate_func=linear)
            self.play(GrowFromEdge(b, LEFT), run_time=2.0, rate_func=linear)
        self.wait(2.0)
        c1 = T("one link busy at a time: HBM idles while C2C works, and back", 22, INK2).to_edge(DOWN, buff=0.75)
        show(self, c1)
        c2 = T("most of the bytes moved are never used", 22, BAD).next_to(c1, DOWN, buff=0.15)
        show(self, c2)
        res = T("~70-80 tok/s at c=1, even with speculative decoding", 26, weight="BOLD").next_to(VGroup(*pts), DOWN, buff=0.35)
        self.play(Write(res), run_time=1.2)
        read(self, res, extra=1.0)
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
        self.wait(1.5)

        attn = VGroup()
        for c in cols:
            b = box(2.4, 0.5, NVL, fill=0.2).move_to(c.get_top() + DOWN * 0.45)
            attn.add(VGroup(b, T("attention: 1/4 heads", 14).move_to(b)))
        self.play(LaggedStart(*[FadeIn(a) for a in attn], lag_ratio=0.15), run_time=1.0)
        self.wait(1.0)
        cap = T("attention is split by heads across the 4 GPUs (TP4)", 22, INK2).to_edge(DOWN, buff=0.35)
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        cells = owned_grids(cols, 1.05)
        grids = VGroup(*cells.values())
        self.play(LaggedStart(*[FadeIn(c) for c in grids], lag_ratio=0.002), run_time=1.5)
        self.wait(1.0)
        cap = T("each GPU owns 96 of the 384 experts of every layer (EP4)", 22, INK2).to_edge(DOWN, buff=0.35)
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        tok0 = [cells[e] for e in STEP["token0"]]
        self.play(*[c.animate.set_fill(C2C, opacity=1) for c in tok0], run_time=0.8)
        self.wait(1.5)
        cap = T("1 token: top-8, so 8 experts per layer", 24, C2C).to_edge(DOWN, buff=0.35)
        show(self, cap, extra=0.5)
        self.play(FadeOut(cap), *[c.animate.set_fill(CELL, opacity=0.9) for c in tok0], run_time=0.6)
        self.wait(0.8)

        act = [cells[e] for e in STEP["active"]]
        self.play(LaggedStart(*[c.animate.set_fill(C2C, opacity=1) for c in act], lag_ratio=0.03), run_time=1.6)
        self.wait(2.0)
        cap = T("verify 8 tokens per step: ~49 experts per layer, up to 64 (8 tokens x top-8)", 22, C2C).to_edge(DOWN, buff=0.35)
        show(self, cap, extra=1.0)
        self.play(FadeOut(cap), run_time=0.4)
        why = [
            T("every weight read now serves up to 8 tokens", 22),
            T("more active experts per step: the hot/cold split and per-GPU load", 20, INK2),
            T("land closer to their averages", 20, INK2),
        ]
        VGroup(*why).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.2)
        show(self, why[0])
        show(self, VGroup(why[1], why[2]), extra=0.8)
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
        self.wait(1.0)
        hot = T("hot experts live in HBM: Marlin reads them at ~2.2 TB/s", 22, HBM).next_to(chip, DOWN, buff=0.35)
        show(self, hot)
        cold = T("cold experts live in Grace RAM: read over C2C at ~420 GB/s", 22, DDR).next_to(hot, DOWN, buff=0.12)
        show(self, cold)

        y1, y2 = -1.25, -2.05
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", DDR, y2)
        note = T("bar length = time", 16, MUTED).next_to(l2, DOWN, buff=0.12).align_to(l2, RIGHT)
        self.play(FadeIn(l1), FadeIn(l2), FadeIn(note), run_time=0.6)
        self.wait(0.8)
        seq = VGroup(seg(-4.2, y1, 2.4, HBM, "hot"), seg(-1.8, y2, 2.4, DDR, "cold"))
        self.play(GrowFromEdge(seq[0], LEFT), run_time=1.4, rate_func=linear)
        self.play(GrowFromEdge(seq[1], LEFT), run_time=1.4, rate_func=linear)
        self.wait(1.2)
        st = T("one after the other: time = hot + cold", 22, BAD).to_edge(DOWN, buff=0.25)
        show(self, st)
        par = VGroup(seg(-4.2, y1, 2.4, HBM, "hot: 5 units of bytes"), seg(-4.2, y2, 2.4, DDR, "cold: 1 unit"))
        self.play(ReplacementTransform(seq, par), FadeOut(st), run_time=1.4)
        self.wait(1.8)
        pt = T("on two streams at once: time = max(hot, cold)", 22, GOOD).to_edge(DOWN, buff=0.25)
        show(self, pt)
        self.play(FadeOut(pt), run_time=0.4)
        ratio = T("both finish together when hot : cold bytes = 2.2 : 0.42  ~  5 : 1", 24, C2C, weight="BOLD").to_edge(DOWN, buff=0.25)
        show(self, ratio, extra=1.0)
        self.play(FadeOut(ratio), run_time=0.4)
        free = T("at that ratio offloading is free, and C2C adds bandwidth on top of HBM", 22, GOOD).to_edge(DOWN, buff=0.25)
        show(self, free, extra=0.8)

        self.play(*[FadeOut(m) for m in (par, l1, l2, note, hot, cold, free)], run_time=0.7)
        better = T("Even better: one kernel that reads from both", 30, weight="BOLD").move_to(UP * 0.2)
        show(self, better)
        sms = VGroup(*[Rectangle(width=0.28, height=0.28, stroke_width=0, fill_color=HBM, fill_opacity=0.85) for _ in range(132)])
        sms.arrange_in_grid(6, 22, buff=0.05).next_to(better, DOWN, buff=0.35)
        lbl = T("132 SMs, one launch", 18, INK2).next_to(sms, DOWN, buff=0.12)
        self.play(LaggedStart(*[FadeIn(s) for s in sms], lag_ratio=0.005), FadeIn(lbl), run_time=1.3)
        self.wait(1.5)
        self.play(VGroup(*sms[:20]).animate.set_fill(DDR), run_time=0.9)
        self.wait(1.5)
        l1 = T("~20 SMs stream cold experts from Grace: enough to saturate C2C", 20, DDR).next_to(lbl, DOWN, buff=0.2)
        show(self, l1)
        l2 = T("the other ~112 stream hot experts from HBM, at the same time", 20, HBM).next_to(l1, DOWN, buff=0.1)
        show(self, l2, extra=1.0)
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
        self.wait(1.0)
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
        self.wait(1.8)
        ww = 1.2 * slow - 1.2
        wait = Rectangle(width=ww, height=0.5, stroke_color=BAD, stroke_width=2, fill_opacity=0).move_to([-4.2 + 1.2 + ww / 2, y1, 0])
        self.play(Create(wait), FadeIn(T("GPU waits", 16, BAD).move_to(wait)), run_time=0.8)
        self.wait(1.5)
        msg = T(f"~{cold_frac:.0%} of the routes on the slow link: cold sets the step time", 23, BAD).to_edge(DOWN, buff=0.3)
        show(self, msg, extra=1.0)
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
        self.wait(1.0)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.004), run_time=3.0)
        self.wait(2.0)
        fair = DashedLine([x0, y0 + unit / n, 0], [x0 + W, y0 + unit / n, 0], color=C2C)
        fl = T("fair share", 16, C2C).next_to(fair, UP, buff=0.05).align_to(fair, RIGHT)
        self.play(Create(fair), FadeIn(fl), run_time=0.8)
        self.wait(1.5)
        c1 = T("a few experts take most of the traffic; the long tail barely runs", 22).move_to([0, 1.55, 0])
        show(self, c1, extra=0.5)
        self.play(FadeOut(c1), run_time=0.4)
        src = VGroup(
            T("profile on a calibration set that looks like deployment", 22, weight="BOLD"),
            T("for us: agentic coding sessions (Claude Code) + autoresearch ML tasks", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.55, 0])
        show(self, src[0])
        show(self, src[1], extra=0.5)
        self.play(FadeOut(src), run_time=0.4)

        half = n // 2
        self.play(*[b.animate.set_fill(HBM) for b in bars[:half]], *[b.animate.set_fill(DDR) for b in bars[half:]], run_time=1.3)
        self.wait(1.8)
        lh = T(f"least-used half -> Grace RAM: only {DATA['cold_share_least_used_half']:.1%} of all routes", 23, DDR).move_to([0, 1.55, 0])
        show(self, lh, extra=0.8)
        self.play(FadeOut(lh), run_time=0.4)
        fin = VGroup(
            T(f"production placement: {DATA['cold_share_profile']:.1%} of routes go to cold experts", 23, GOOD, weight="BOLD"),
            T("within the ~5:1 the overlap can hide", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to([0, 1.55, 0])
        show(self, fin[0])
        show(self, fin[1], extra=1.0)
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
        self.wait(1.5)

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
        self.play(now.animate.set_value(t1), run_time=9, rate_func=linear)
        self.remove(cursor)
        self.wait(2.0)

        cap_y = -3.45
        c1 = T("each GPU reads its hot and cold experts at the same time", 21).move_to([0, cap_y, 0])
        show(self, c1)
        self.play(FadeOut(c1), run_time=0.4)
        hl = VGroup(SurroundingRectangle(VGroup(segs[6], segs[10]), color=DDR, buff=0.06))
        self.play(Create(hl), run_time=0.8)
        self.wait(1.0)
        c2 = T("GPUs 1 and 2 drew lots of cold experts: cold sets their pace", 21, DDR).move_to([0, cap_y, 0])
        show(self, c2)
        self.play(FadeOut(c2), FadeOut(hl), run_time=0.4)
        hl = VGroup(SurroundingRectangle(segs[3], color=WAIT, buff=0.06), SurroundingRectangle(segs[15], color=WAIT, buff=0.06))
        self.play(Create(hl), run_time=0.8)
        self.wait(1.0)
        c3 = T("GPUs 0 and 3 finish early and sit in the collective, ~150 us doing nothing", 21, WAIT).move_to([0, cap_y, 0])
        show(self, c3, extra=0.5)
        self.play(FadeOut(c3), FadeOut(hl), run_time=0.4)
        c4 = T("which GPU is last changes every layer: routing luck, not a slow GPU", 21).move_to([0, cap_y, 0])
        show(self, c4, extra=1.0)
        clear(self)


class S9Replicas(Scene):
    def construct(self):
        title(self, "Idea 3: replicas for the cold experts", "back to the per-GPU layout")
        cols = gpu_columns(4.3).shift(DOWN * 0.75)
        heads = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").next_to(c, UP, buff=0.08) for i, c in enumerate(cols)])
        cells = owned_grids(cols, 0.5)
        self.play(FadeIn(cols), FadeIn(heads), FadeIn(VGroup(*cells.values())), run_time=1.0)
        self.wait(1.0)
        cap_y = 2.35

        # Split each GPU's experts into an HBM panel and a Grace panel.
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
        self.play(FadeIn(panels), FadeIn(plabels), run_time=0.7)
        self.play(*moves, run_time=1.8)
        self.wait(1.2)
        cap = T("after placement: hot experts in HBM, cold ones in that GPU's Grace RAM", 21, INK2).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        # The real verify step.
        cold_act = set(STEP["cold_active"])
        hot_anims = [cells[e].animate.set_fill(INK, opacity=1) for e in STEP["active"] if e not in cold_act]
        cold_anims = [cells[e].animate.set_fill(DDR, opacity=1).set_stroke(INK, 1.5) for e in cold_act]
        self.play(*hot_anims, run_time=0.9)
        self.wait(1.0)
        self.play(*cold_anims, run_time=0.9)
        self.wait(1.5)
        cap = T(f"one real verify step (layer {STEP['layer']}): {len(STEP['active'])} experts active, {len(cold_act)} of them cold", 21).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        def counters(vals, worst_color):
            m = max(vals)
            return VGroup(*[
                T(f"cold reads: {v}", 18, worst_color if v == m else INK2, weight="BOLD" if v == m else "NORMAL").next_to(cols[i], DOWN, buff=0.12)
                for i, v in enumerate(vals)
            ])
        before = counters(STEP["no_rep"], BAD)
        self.play(FadeIn(before), run_time=0.7)
        self.wait(1.5)
        worst = int(np.argmax(STEP["no_rep"]))
        self.play(Indicate(cols[worst], color=BAD, scale_factor=1.03), run_time=1.0)
        cap = T(f"the layer waits for GPU {worst} and its {max(STEP['no_rep'])} cold reads", 21, BAD).move_to([0, cap_y, 0])
        show(self, cap, extra=0.5)
        self.play(FadeOut(cap), run_time=0.4)

        # Replicas: copies of other GPUs' cold experts in spare Grace RAM.
        rep_cells, rep_anims = {}, []
        for r in range(4):
            info = STEP["ranks"][r]
            base = len(info["cold"])
            for k, e in enumerate(info["replicas"]):
                c = Rectangle(width=CELL_W, height=CELL_W, stroke_color=DDR, stroke_width=1.2, fill_color=DDR, fill_opacity=0.12)
                c.move_to(grid_pos(origins[r][1], base + k))
                rep_cells[(e, r)] = c
                rep_anims.append(FadeIn(c))
        self.play(LaggedStart(*rep_anims, lag_ratio=0.01), run_time=1.3)
        self.wait(1.2)
        cap = T("spare Grace RAM holds copies (outlined) of other GPUs' cold experts", 21).move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)

        moved = [(int(e), STEP["owner"][e], r) for e, r in STEP["assigned"].items() if STEP["owner"][e] != r]
        arrows, anims = VGroup(), []
        for e, o, r in moved:
            src, dst = cells[e], rep_cells[(e, r)]
            arrows.add(Arrow(src.get_center(), dst.get_center(), buff=0.05, stroke_width=3, color=C2C,
                             max_tip_length_to_length_ratio=0.08))
            anims += [src.animate.set_fill(DDR, opacity=0.35).set_stroke(width=0),
                      dst.animate.set_fill(DDR, opacity=1).set_stroke(INK, 1.5)]
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.25), run_time=1.6)
        self.play(*anims, run_time=0.9)
        self.wait(1.2)
        cap = T("each step, every cold expert is read from the copy on the less-loaded GPU", 21).move_to([0, cap_y, 0])
        show(self, cap)
        after = counters(STEP["with_rep"], GOOD)
        self.play(FadeOut(arrows), ReplacementTransform(before, after), FadeOut(cap), run_time=1.2)
        self.wait(1.5)
        cap = T(f"this layer now waits for {max(STEP['with_rep'])} cold reads instead of {max(STEP['no_rep'])}", 22, GOOD, weight="BOLD").move_to([0, cap_y, 0])
        show(self, cap)
        self.play(FadeOut(cap), run_time=0.4)
        stats = T(
            f"over all steps: busiest GPU {DATA['max_cold_rank_no_replicas_mean']:.2f} -> "
            f"{DATA['max_cold_rank_replicas_mean']:.2f} cold reads (perfect balance {DATA['ideal_max_cold_rank_mean']:.2f})",
            20, INK2,
        ).move_to([0, cap_y, 0])
        show(self, stats, extra=1.5)
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
        self.wait(1.8)
        self.play(GrowFromEdge(b, LEFT), FadeIn(bl), run_time=1.8)
        self.wait(5)
