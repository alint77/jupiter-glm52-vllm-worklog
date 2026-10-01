"""Tiered MoE on a 4x GH200 node, explained with MiMo-V2.6-Pro.

Render one scene:   manim -qh tiered_moe.py S1Hardware
Render all + join:  bash render.sh [l|m|h]

Numbers come from the JSC JUPITER configuration page, the MiMo-V2.6-Pro
checkpoint, data.json (real routing traces, see prep_data.py) and measured
serving results.
"""

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UL,
    UR,
    DL,
    DR,
    AnimationGroup,
    Arrow,
    Create,
    DashedLine,
    DoubleArrow,
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
    VGroup,
    Write,
    config,
    linear,
    smooth,
)

DATA = json.loads((Path(__file__).parent / "data.json").read_text())

BG = "#101114"
INK = "#ECEBE6"
INK2 = "#A9A8A2"
MUTED = "#6E6D68"
HBM = "#3D8BEA"  # VRAM / hot
DDR = "#F07A3A"  # Grace RAM / cold
C2C = "#E9C46A"
NVL = "#7BC8A4"
BAD = "#E05B5B"
GOOD = "#7BC8A4"
FONT = "DejaVu Sans"

config.background_color = BG


def T(text, size=28, color=INK, weight="NORMAL", **kw):
    return Text(text, font=FONT, font_size=size, color=color, weight=weight, disable_ligatures=True, **kw)


def title(scene, text, sub=None):
    t = T(text, 40, weight="BOLD").to_edge(UP, buff=0.4)
    group = VGroup(t)
    if sub:
        s = T(sub, 24, INK2).next_to(t, DOWN, buff=0.15)
        group.add(s)
    scene.play(FadeIn(group, shift=DOWN * 0.2), run_time=0.8)
    return group


def box(w, h, color, label=None, size=22, fill=0.18):
    r = RoundedRectangle(
        corner_radius=0.12, width=w, height=h, stroke_color=color,
        stroke_width=3, fill_color=color, fill_opacity=fill,
    )
    if label is None:
        return r
    lab = T(label, size, INK)
    if lab.width > w * 0.9:
        lab.scale_to_fit_width(w * 0.9)
    return VGroup(r, lab.move_to(r))


def clear(scene, run_time=0.6):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


def superchip(scale=1.0, mirror=False):
    """Grace + Hopper joined by NVLink-C2C; mirror puts Hopper on the left."""
    grace = VGroup(
        box(2.3, 1.55, DDR),
        T("Grace CPU", 22, weight="BOLD"),
        T("72 cores", 17, INK2),
        T("120 GB LPDDR5X", 17, INK2),
        T("512 GB/s", 17, INK2),
    )
    grace[1:].arrange(DOWN, buff=0.08).move_to(grace[0])
    hopper = VGroup(
        box(2.3, 1.55, HBM),
        T("Hopper GPU", 22, weight="BOLD"),
        T("132 SMs", 17, INK2),
        T("96 GB HBM3", 17, INK2),
        T("4 TB/s", 17, INK2),
    )
    hopper[1:].arrange(DOWN, buff=0.08).move_to(hopper[0])
    hopper.next_to(grace, LEFT if mirror else RIGHT, buff=0.9)
    a, b = (grace.get_left(), hopper.get_right()) if mirror else (grace.get_right(), hopper.get_left())
    link = Line(a, b, color=C2C, stroke_width=10)
    lab = T("C2C", 16, C2C).next_to(link, UP, buff=0.06)
    chip = VGroup(grace, hopper, link, lab)
    chip.grace, chip.hopper, chip.link = grace, hopper, link
    return chip.scale(scale)


# --------------------------------------------------------------------------
class S1Hardware(Scene):
    def construct(self):
        head = title(self, "The machine: one JUPITER Booster node", "4x NVIDIA GH200 Grace-Hopper superchips")

        one = superchip(1.25).move_to(DOWN * 0.3)
        self.play(FadeIn(one.grace, shift=RIGHT * 0.2), FadeIn(one.hopper, shift=LEFT * 0.2))
        self.play(Create(one.link), FadeIn(one[3]))
        c2c = T("NVLink-C2C: 900 GB/s  (450 GB/s each way)", 26, C2C).next_to(one, DOWN, buff=0.5)
        self.play(Write(c2c))
        note = T("the GPU can read CPU memory directly, ~8x faster than PCIe", 22, INK2).next_to(c2c, DOWN, buff=0.2)
        self.play(FadeIn(note))
        self.wait(1.5)
        self.play(FadeOut(c2c), FadeOut(note))

        # Four superchips in a 2x2 grid.
        chips = VGroup(*[superchip(0.62, mirror=(i % 2 == 1)) for i in range(4)])
        chips.arrange_in_grid(2, 2, buff=(1.3, 1.1)).move_to(DOWN * 0.35)
        self.play(ReplacementTransform(one, chips[0]), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(c) for c in chips[1:]], lag_ratio=0.2))
        idx = VGroup(*[T(f"GPU {i}", 16, INK2).next_to(c.hopper, UP, buff=0.06) for i, c in enumerate(chips)])
        self.play(FadeIn(idx))

        gpus = [c.hopper for c in chips]
        cpus = [c.grace for c in chips]
        nvl = VGroup()
        for i in range(4):
            for j in range(i + 1, 4):
                nvl.add(Line(gpus[i].get_center(), gpus[j].get_center(), color=NVL, stroke_width=3).set_z_index(-1))
        nvl_lab = T("NVLink 4, every GPU pair: 150 GB/s each way", 22, NVL).to_edge(DOWN, buff=0.7)
        self.play(Create(nvl), Write(nvl_lab))
        cpu_links = VGroup(
            Line(cpus[0].get_bottom(), cpus[2].get_top(), color=DDR, stroke_width=2),
            Line(cpus[1].get_bottom(), cpus[3].get_top(), color=DDR, stroke_width=2),
        )
        cpu_lab = T("CPU-CPU: 100 GB/s each way    |    network: 4x InfiniBand NDR200 (25 GB/s each)", 20, INK2).to_edge(DOWN, buff=0.3)
        self.play(Create(cpu_links), FadeIn(cpu_lab))
        self.wait(2.5)

        # Totals.
        node = VGroup(chips, idx)
        self.play(FadeOut(nvl), FadeOut(cpu_links), FadeOut(nvl_lab), FadeOut(cpu_lab), node.animate.scale(0.78).to_edge(LEFT, buff=0.3))
        rows = VGroup(
            T("Per node", 28, weight="BOLD"),
            T("384 GB HBM3   (4 x 96)", 24, HBM),
            T("480 GB LPDDR5X (4 x 120)", 24, DDR),
            T("~990 TFLOPS BF16 per GPU", 24, INK2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.5).shift(UP * 1.5)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows], lag_ratio=0.25))

        # Bandwidth ladder.
        ladder = [("HBM -> GPU", 4000, HBM), ("Grace RAM -> GPU (C2C)", 450, C2C),
                  ("GPU <-> GPU (NVLink)", 150, NVL), ("node <-> node (IB)", 25, BAD)]
        bars = VGroup()
        for name, gbs, col in ladder:
            w = max(0.05, 4.2 * gbs / 4000)
            b = Rectangle(width=w, height=0.28, fill_color=col, fill_opacity=0.9, stroke_width=0)
            lab = T(f"{name}  {gbs if gbs < 1000 else '4,000'} GB/s", 18, INK).next_to(b, UP, buff=0.05, aligned_edge=LEFT)
            bars.add(VGroup(lab, b))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.16).next_to(rows, DOWN, buff=0.4, aligned_edge=LEFT)
        for g in bars:
            g[1].align_to(bars, LEFT)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT)) for g in bars], lag_ratio=0.3), run_time=2.2)
        key = T("C2C: the second-fastest path into the GPU", 20, C2C).next_to(bars, DOWN, buff=0.25, aligned_edge=LEFT)
        if key.get_right()[0] > 6.9:
            key.shift(LEFT * (key.get_right()[0] - 6.9))
        self.play(Write(key), Indicate(bars[1][1], color=C2C))
        self.wait(3)
        clear(self)


# --------------------------------------------------------------------------
class S2Problem(Scene):
    def construct(self):
        title(self, "The model doesn't fit", "MiMo-V2.6-Pro: ~1T params, MXFP4")
        facts = VGroup(
            T("69 MoE layers x 384 experts, top-8 routing", 26),
            T("checkpoint: 566 GB, nearly all of it experts", 26),
        ).arrange(DOWN, buff=0.2).shift(UP * 1.3)
        self.play(FadeIn(facts, shift=UP * 0.2))

        unit = 6.0 / 566
        model = Rectangle(width=566 * unit, height=0.6, fill_color=INK2, fill_opacity=0.8, stroke_width=0)
        hbm = Rectangle(width=384 * unit, height=0.6, fill_color=HBM, fill_opacity=0.9, stroke_width=0)
        VGroup(model, hbm).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(DOWN * 0.3 + RIGHT * 0.3)
        ml = T("model weights 566 GB", 22).next_to(model, LEFT, buff=0.3)
        hl = T("node HBM 384 GB", 22, HBM).next_to(hbm, LEFT, buff=0.3)
        self.play(GrowFromEdge(model, LEFT), FadeIn(ml))
        self.play(GrowFromEdge(hbm, LEFT), FadeIn(hl))
        gap = DashedLine(hbm.get_right() + UP * 1.2, hbm.get_right() + DOWN * 0.4, color=BAD)
        over = T("+ KV cache, activations...", 20, BAD).next_to(model, RIGHT, buff=0.2)
        self.play(Create(gap), FadeIn(over))
        self.wait(1)

        opts = VGroup(
            VGroup(T("Option A: two nodes", 26, weight="BOLD"),
                   T("8 GPUs, but every layer's collectives", 20, INK2),
                   T("cross InfiniBand at 25 GB/s per GPU", 20, INK2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
            VGroup(T("Option B: offload to Grace RAM", 26, weight="BOLD"),
                   T("480 GB sitting right next to the GPUs,", 20, INK2),
                   T("reachable at 450 GB/s over C2C", 20, INK2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
        ).arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(opts[0], shift=UP * 0.2))
        self.play(FadeIn(opts[1], shift=UP * 0.2))
        self.play(opts[0].animate.set_opacity(0.35), Indicate(opts[1], color=DDR, scale_factor=1.05))
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
        lab = T(text, 18, BG if color != MUTED else INK).move_to(r)
        if lab.width > w * 0.92:
            lab.scale_to_fit_width(w * 0.92)
        return VGroup(r, lab)
    return r


class S3StockVllm(Scene):
    def construct(self):
        title(self, "How stock vLLM offloads", "--cpu-offload-gb")
        pts = VGroup(
            T("moves whole layers' weights to CPU memory", 24),
            T("every forward pass streams all of them back over the link", 24),
            T("including the experts this token never routes to", 24, BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).shift(UP * 1.6)
        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in pts], lag_ratio=0.4), run_time=2)

        y1, y2 = -0.6, -1.4
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", C2C, y2)
        self.play(FadeIn(l1), FadeIn(l2))
        x = -4.2
        blocks = VGroup()
        for i in range(2):
            a = seg(x, y1, 1.3, HBM, "HBM layer")
            x += 1.3
            b = seg(x, y2, 3.0, C2C, "offloaded layer: all experts")
            x += 3.0
            blocks.add(a, b)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in blocks], lag_ratio=0.6), run_time=3.5)
        idle = VGroup(
            T("one link busy at a time: HBM idles while C2C works and vice versa", 22, INK2),
            T("and most of the bytes moved are never used", 22, BAD),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(idle))
        res = T("~70-80 tok/s at c=1, even with speculative decoding", 26, weight="BOLD").next_to(pts, DOWN, buff=0.35)
        self.play(Write(res))
        self.wait(3)
        clear(self)


# --------------------------------------------------------------------------
class S4Layout(Scene):
    def construct(self):
        title(self, "Our layout: TP4 attention, EP4 experts", "plus speculative decoding: verify 8 tokens per step")
        gpus = VGroup(*[box(2.4, 2.9, HBM) for _ in range(4)]).arrange(RIGHT, buff=0.35).shift(DOWN * 0.3)
        heads = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").next_to(g, UP, buff=0.1) for i, g in enumerate(gpus)])
        self.play(FadeIn(gpus), FadeIn(heads))
        attn = VGroup(*[box(2.1, 0.5, NVL, "attention: 1/4 heads", 16).move_to(g.get_top() + DOWN * 0.45) for g in gpus])
        self.play(LaggedStart(*[FadeIn(a) for a in attn], lag_ratio=0.15))
        exp = VGroup()
        for gi, g in enumerate(gpus):
            grid = VGroup(*[Rectangle(width=0.16, height=0.16, stroke_width=0, fill_color=INK2, fill_opacity=0.7) for _ in range(96)])
            grid.arrange_in_grid(8, 12, buff=0.03).move_to(g.get_center() + DOWN * 0.2)
            exp.add(grid)
        lab = VGroup(*[T(f"experts {96*i}-{96*i+95}", 15, INK2).next_to(exp[i], DOWN, buff=0.08) for i in range(4)])
        self.play(LaggedStart(*[FadeIn(e) for e in exp], lag_ratio=0.15), FadeIn(lab))
        self.wait(0.8)

        # Speculative decoding: 1 token vs 8 tokens.
        rng = np.random.default_rng(3)
        def light(n_tok):
            ids = set()
            for _ in range(n_tok):
                ids.update(rng.choice(384, 8, replace=False).tolist())
            anims = []
            for e in ids:
                anims.append(exp[e // 96][e % 96].animate.set_fill(C2C, opacity=1.0))
            return ids, anims
        cap = T("1 token: 8 experts active per layer", 24, C2C).to_edge(DOWN, buff=0.4)
        ids, anims = light(1)
        self.play(*anims, FadeIn(cap), run_time=0.8)
        self.wait(1)
        reset = [exp[e // 96][e % 96].animate.set_fill(INK2, opacity=0.7) for e in ids]
        self.play(*reset, run_time=0.4)
        cap2 = T(f"verify 8 tokens: ~{DATA['active_per_layer_8tok_mean']:.0f} experts active per layer (real traces)", 24, C2C).to_edge(DOWN, buff=0.4)
        ids, anims = light(8)
        self.play(*anims, ReplacementTransform(cap, cap2), run_time=1.0)
        self.wait(0.8)
        why = VGroup(
            T("more active experts per step:", 22, weight="BOLD"),
            T("each weight read serves up to 8 tokens", 20, INK2),
            T("hot/cold split and per-GPU load land closer to their averages", 20, INK2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_corner(DR, buff=0.3).shift(UP * 0.5)
        self.play(FadeOut(cap2), FadeIn(why, shift=UP * 0.2))
        self.wait(3.5)
        clear(self)


# --------------------------------------------------------------------------
class S5Overlap(Scene):
    def construct(self):
        title(self, "Idea 1: read both memories at once", "decode is bandwidth-bound")
        chip = superchip(0.9).to_edge(UP, buff=1.5)
        hot = T("hot experts in HBM: ~2.2 TB/s (Marlin)", 20, HBM)
        cold = T("cold experts in Grace RAM: ~420 GB/s (C2C)", 20, DDR)
        VGroup(hot, cold).arrange(RIGHT, buff=0.5).next_to(chip, DOWN, buff=0.3)
        if VGroup(hot, cold).width > 13.4:
            VGroup(hot, cold).scale_to_fit_width(13.4)
        self.play(FadeIn(chip), FadeIn(hot), FadeIn(cold))

        y1, y2 = -1.3, -2.1
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", DDR, y2)
        self.play(FadeIn(l1), FadeIn(l2))
        seq = VGroup(seg(-4.2, y1, 2.5, HBM, "hot"), seg(-1.7, y2, 2.5, DDR, "cold"))
        st = T("one after the other: hot + cold", 22, BAD).to_edge(DOWN, buff=0.25)
        self.play(GrowFromEdge(seq[0], LEFT), run_time=1.2, rate_func=linear)
        self.play(GrowFromEdge(seq[1], LEFT), FadeIn(st), run_time=1.2, rate_func=linear)
        self.wait(0.6)
        par = VGroup(seg(-4.2, y1, 2.5, HBM, "hot  5 units of bytes"), seg(-4.2, y2, 2.5, DDR, "cold  1 unit"))
        pt = T("on two streams: max(hot, cold) - offloading is free, or faster", 22, GOOD).to_edge(DOWN, buff=0.25)
        self.play(ReplacementTransform(seq, par), ReplacementTransform(st, pt), run_time=1.2)
        ratio = T("balanced when hot : cold bytes = 2.2 TB/s : 0.42 TB/s  ~  5 : 1", 26, C2C, weight="BOLD").shift(DOWN * 0.45)
        note = T("(bar length = time)", 16, MUTED).next_to(l2, DOWN, buff=0.12).align_to(l2, RIGHT)
        self.play(Write(ratio), FadeIn(note))
        self.wait(3)

        # Better: one kernel.
        self.play(FadeOut(par), FadeOut(pt), FadeOut(l1), FadeOut(l2), FadeOut(ratio), FadeOut(note), FadeOut(hot), FadeOut(cold))
        better = T("Better: one kernel that reads from both", 30, weight="BOLD").shift(UP * 0.1)
        self.play(FadeIn(better))
        sms = VGroup(*[Rectangle(width=0.28, height=0.28, stroke_width=0, fill_color=HBM, fill_opacity=0.85) for _ in range(132)])
        sms.arrange_in_grid(6, 22, buff=0.05).next_to(better, DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(s) for s in sms], lag_ratio=0.005), run_time=1.2)
        cold_sms = VGroup(*sms[:20])
        self.play(cold_sms.animate.set_fill(DDR), run_time=0.8)
        legend = VGroup(
            T("~20 SMs stream cold experts from Grace: enough to saturate C2C", 20, DDR),
            T("the other ~112 stream hot experts from HBM, in the same launch", 20, HBM),
        ).arrange(DOWN, buff=0.1).next_to(sms, DOWN, buff=0.3)
        self.play(FadeIn(legend))
        self.wait(3.5)
        clear(self)


# --------------------------------------------------------------------------
class S6HalfOffloaded(Scene):
    def construct(self):
        title(self, "The catch: we must offload a lot", "566 GB of weights, 384 GB of HBM")
        cold_frac = 1 - DATA["hot_fraction_of_experts"]
        bar = VGroup(
            Rectangle(width=10 * (1 - cold_frac), height=0.6, fill_color=HBM, fill_opacity=0.9, stroke_width=0),
            Rectangle(width=10 * cold_frac, height=0.6, fill_color=DDR, fill_opacity=0.9, stroke_width=0),
        ).arrange(RIGHT, buff=0).shift(UP * 1.2)
        labs = VGroup(
            T(f"{1-cold_frac:.0%} of experts in HBM", 20).next_to(bar[0], DOWN, buff=0.12),
            T(f"{cold_frac:.0%} in Grace RAM", 20).next_to(bar[1], DOWN, buff=0.12),
        )
        self.play(GrowFromEdge(bar, LEFT), FadeIn(labs))
        q = T("if every expert were used equally often...", 26).shift(DOWN * 0.05)
        self.play(FadeIn(q))
        y1, y2 = -1.3, -2.1
        l1, l2 = lane("HBM", HBM, y1), lane("C2C", DDR, y2)
        slow = (cold_frac / 0.42) / ((1 - cold_frac) / 2.2)
        hotb = seg(-4.2, y1, 1.2, HBM, "hot")
        coldb = seg(-4.2, y2, 1.2 * slow, DDR, f"cold: ~{slow:.0f}x longer than hot")
        self.play(FadeIn(l1), FadeIn(l2))
        self.play(GrowFromEdge(hotb, LEFT), GrowFromEdge(coldb, LEFT), run_time=2, rate_func=linear)
        ww = 1.2 * slow - 1.2
        wait = Rectangle(width=ww, height=0.5, stroke_color=BAD, stroke_width=2, fill_opacity=0).move_to([-4.2 + 1.2 + ww / 2, y1, 0])
        wl = T("GPU waits", 16, BAD).move_to(wait)
        self.play(Create(wait), FadeIn(wl))
        msg = T(f"~{cold_frac:.0%} of routes on the slow link: cold sets the step time", 24, BAD).to_edge(DOWN, buff=0.3)
        self.play(Write(msg))
        self.wait(3)
        clear(self)


# --------------------------------------------------------------------------
class S7Frequency(Scene):
    def construct(self):
        title(self, "Idea 2: offload the experts nobody uses", "routing is far from uniform")
        share = np.array(DATA["sorted_share_layer"])
        n = len(share)
        W, H = 10.5, 3.6
        x0, y0 = -W / 2, -2.3
        unit = H / share.max()
        bars = VGroup()
        for i, s in enumerate(share):
            r = Rectangle(width=W / n, height=max(0.005, s * unit), stroke_width=0, fill_color=INK2, fill_opacity=0.9)
            r.move_to([x0 + (i + 0.5) * W / n, y0, 0], aligned_edge=DOWN)
            bars.add(r)
        axis = Line([x0, y0, 0], [x0 + W, y0, 0], color=MUTED)
        xl = T(f"384 experts of layer {DATA['layer_shown']}, sorted by how often they are routed to", 18, INK2).next_to(axis, DOWN, buff=0.12)
        self.play(Create(axis), FadeIn(xl))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.004), run_time=2.5)
        fair = DashedLine([x0, y0 + unit / n, 0], [x0 + W, y0 + unit / n, 0], color=C2C)
        fl = T("fair share", 16, C2C).next_to(fair, UP, buff=0.05).align_to(fair, RIGHT)
        self.play(Create(fair), FadeIn(fl))
        self.wait(0.8)

        src = VGroup(
            T("profile on a calibration set that looks like deployment:", 22, weight="BOLD"),
            T("our agentic coding sessions (Claude Code) + autoresearch ML tasks", 20, INK2),
        ).arrange(DOWN, buff=0.1).next_to(VGroup(bars), UP, buff=0.25)
        self.play(FadeIn(src))
        self.wait(1)

        half = n // 2
        self.play(*[b.animate.set_fill(HBM) for b in bars[:half]], *[b.animate.set_fill(DDR) for b in bars[half:]], run_time=1.2)
        lh = T(f"least-used half -> Grace: only {DATA['cold_share_least_used_half']:.1%} of all routes", 22, DDR).move_to([x0 + 0.68 * W, y0 + 1.8, 0])
        self.play(FadeIn(lh))
        self.wait(2.5)
        fin = VGroup(
            T(f"production placement: {DATA['cold_share_profile']:.1%} of routes go cold", 24, GOOD, weight="BOLD"),
            T("better than the 5:1 overlap needs, so cold hides behind hot", 20, INK2),
        ).arrange(DOWN, buff=0.1).move_to([x0 + 0.68 * W, y0 + 1.8, 0])
        self.play(FadeOut(lh), FadeIn(fin))
        self.wait(3.5)
        clear(self)


# --------------------------------------------------------------------------
class S8Replicas(Scene):
    def construct(self):
        title(self, "Idea 3: replicas for the cold experts", "the slowest GPU sets the step time")
        ex = DATA["example_step"]
        no_rep, with_rep = ex["no_rep"], ex["with_rep"]
        cols = VGroup(*[VGroup() for _ in range(4)])
        base_y = -2.2
        xs = [-4.5, -1.5, 1.5, 4.5]
        heads = VGroup(*[T(f"GPU {i}", 22, weight="BOLD").move_to([xs[i], base_y - 0.4, 0]) for i in range(4)])
        ground = Line([-6, base_y, 0], [6, base_y, 0], color=MUTED)
        self.play(Create(ground), FadeIn(heads))
        cap = T(f"one real verify step, layer {ex['layer']}: cold experts each GPU must read", 22, INK2).shift(UP * 2.0)
        self.play(FadeIn(cap))

        def stack(counts):
            g = VGroup()
            for i, c in enumerate(counts):
                col = VGroup()
                for k in range(c):
                    r = Rectangle(width=1.6, height=0.42, stroke_width=1.5, stroke_color=BG, fill_color=DDR, fill_opacity=0.9)
                    r.move_to([xs[i], base_y + 0.21 + 0.44 * k, 0])
                    col.add(r)
                g.add(col)
            return g

        before = stack(no_rep)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.3) for c in before], lag_ratio=0.2))
        top = base_y + 0.44 * max(no_rep)
        line = DashedLine([-6, top, 0], [6, top, 0], color=BAD)
        wl = T(f"everyone waits for GPU {int(np.argmax(no_rep))}: {max(no_rep)} cold reads", 20, BAD).next_to(line, UP, buff=0.08).to_edge(RIGHT, buff=0.6)
        self.play(Create(line), FadeIn(wl))
        self.wait(1.2)

        how = VGroup(
            T("spare Grace RAM holds a second copy of cold experts on another GPU", 20),
            T("per step, the router sends each cold expert to the less-loaded copy", 20),
        ).arrange(DOWN, buff=0.08).next_to(cap, DOWN, buff=0.15)
        self.play(FadeIn(how))
        after = stack(with_rep)
        top2 = base_y + 0.44 * max(with_rep)
        line2 = DashedLine([-6, top2, 0], [6, top2, 0], color=GOOD)
        wl2 = T(f"now the step waits for {max(with_rep)}", 20, GOOD).next_to(line2, UP, buff=0.08).to_edge(RIGHT, buff=0.6)
        self.play(ReplacementTransform(before, after), ReplacementTransform(line, line2), ReplacementTransform(wl, wl2), run_time=1.6)
        self.wait(1)
        stats = T(
            f"over all steps: busiest GPU {DATA['max_cold_rank_no_replicas_mean']:.2f} -> "
            f"{DATA['max_cold_rank_replicas_mean']:.2f} cold reads (perfect balance: {DATA['ideal_max_cold_rank_mean']:.2f})",
            20, INK2,
        ).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(stats))
        self.wait(3.5)
        clear(self)


# --------------------------------------------------------------------------
class S9Result(Scene):
    def construct(self):
        title(self, "Putting it together", "MiMo-V2.6-Pro, 4x GH200, one user (c=1)")
        items = VGroup(
            T("1. hot and cold experts read in parallel, in one kernel", 24),
            T("2. placement from a deployment-like calibration profile", 24),
            T("3. replicas balance cold work across GPUs", 24),
            T("prefill: copy layer N+1's cold experts into HBM during layer N (double buffered)", 20, INK2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(UP * 0.8)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in items], lag_ratio=0.35), run_time=2.5)
        unit = 5.0 / 210
        a = Rectangle(width=75 * unit, height=0.5, fill_color=MUTED, fill_opacity=0.9, stroke_width=0)
        b = Rectangle(width=210 * unit, height=0.5, fill_color=GOOD, fill_opacity=0.9, stroke_width=0)
        VGroup(a, b).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 2.0 + RIGHT * 1.6)
        al = T("stock vLLM offload + SD   ~70-80 tok/s", 20).next_to(a, LEFT, buff=0.3)
        bl = T("tiered MoE   ~210 tok/s", 20, GOOD, weight="BOLD").next_to(b, LEFT, buff=0.3)
        self.play(GrowFromEdge(a, LEFT), FadeIn(al))
        self.play(GrowFromEdge(b, LEFT), FadeIn(bl), run_time=1.5)
        self.wait(4)
