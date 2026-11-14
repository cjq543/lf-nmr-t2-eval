# -*- coding: utf-8 -*-
"""
Regenerate the simplified 2.5:1 (2500 x 1000 px, 300 dpi) Graphical Abstract
inside the standalone public ReproBundle (outputs to results/regenerated_figures/).

Design adheres strictly to Elsevier's Graphical Abstract guidelines (500:200 aspect
ratio, minimal dense text, large high-contrast labels readable down to 500 x 200 px)
while preserving exact verified manuscript numbers and separating 16.99 s file-replay
harness time from 14.04 s vs. 0.179 s fixed-input compute benchmarks.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "regenerated_figures"


def _load_font(candidates, size: int, bold: bool = False):
    search_dirs = [
        Path(r"C:\Windows\Fonts"),
        Path("/usr/share/fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]
    for name in candidates:
        for d in search_dirs:
            p = d / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    try:
        from matplotlib import font_manager
        fp = font_manager.findfont(
            font_manager.FontProperties(
                family="DejaVu Sans",
                weight="bold" if bold else "normal",
            )
        )
        return ImageFont.truetype(fp, size)
    except Exception:
        return ImageFont.load_default()


def _centered(draw, cx: float, y: float, text: str, font, fill):
    bb = draw.textbbox((0, 0), text, font=font)
    w = bb[2] - bb[0]
    draw.text((cx - w / 2.0, y), text, font=font, fill=fill)
    return bb


def _fit_text(draw, box_x0: float, box_x1: float, y: float, text: str, font, fill, align="center"):
    bb = draw.textbbox((0, 0), text, font=font)
    w = bb[2] - bb[0]
    avail = box_x1 - box_x0
    assert w <= avail, f"GA text overflow ({w:.0f} > {avail:.0f}): {text!r}"
    if align == "center":
        x = (box_x0 + box_x1 - w) / 2.0
    elif align == "right":
        x = box_x1 - w
    else:
        x = box_x0
    draw.text((x, y), text, font=font, fill=fill)


def build_ga_image() -> Image.Image:
    S = 2
    W, H = 2500 * S, 1000 * S
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    reg_c = ["arial.ttf", "segoeui.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"]
    bold_c = ["arialbd.ttf", "segoeuib.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]

    f_hdr = _load_font(bold_c, 31 * S, bold=True)
    f_sec = _load_font(bold_c, 27 * S, bold=True)
    f_val = _load_font(bold_c, 29 * S, bold=True)
    f_badge = _load_font(bold_c, 25 * S, bold=True)
    f_lbl = _load_font(bold_c, 24 * S, bold=True)
    f_sub = _load_font(reg_c, 23 * S, bold=False)
    f_tick = _load_font(reg_c, 22 * S, bold=False)

    NAVY = (22, 43, 76)
    SLATE = (71, 85, 105)
    BORDER = (196, 206, 220)
    HDR_BG = (234, 242, 250)
    BLUE = (28, 99, 168)
    BLUE_LT = (232, 243, 253)
    ORANGE = (214, 92, 12)
    ORANGE_LT = (254, 241, 228)
    RED = (194, 44, 44)
    TEAL = (19, 122, 107)
    TEAL_LT = (230, 246, 243)
    GRID = (220, 228, 238)

    top_y, bot_y = 22 * S, 978 * S
    hdr_h = 68 * S
    panels = [
        (24 * S, 816 * S, "1. Observation Window & Loss"),
        (842 * S, 1658 * S, "2. Protocol Flips Ranking (n = 35)"),
        (1684 * S, 2476 * S, "3. Denoising & Zynq-7020 Cost"),
    ]
    for x0, x1, title in panels:
        draw.rounded_rectangle((x0, top_y, x1, bot_y), radius=18 * S, fill=(255, 255, 255), outline=BORDER, width=2 * S)
        draw.rounded_rectangle((x0, top_y, x1, top_y + hdr_h), radius=18 * S, fill=HDR_BG, outline=BORDER, width=2 * S)
        draw.rectangle((x0 + 2 * S, top_y + hdr_h - 18 * S, x1 - 2 * S, top_y + hdr_h), fill=HDR_BG)
        draw.line([(x0, top_y + hdr_h), (x1, top_y + hdr_h)], fill=BORDER, width=2 * S)
        _fit_text(draw, x0 + 16 * S, x1 - 16 * S, top_y + 17 * S, title, f_hdr, NAVY, align="center")

    # =========================================================================
    # PANEL 1: Observation Window Schematic (top) + Bar Comparison (bottom)
    # =========================================================================
    p1_x0, p1_x1 = panels[0][0], panels[0][1]
    ax0, ay0, ax1, ay1 = p1_x0 + 84 * S, top_y + 102 * S, p1_x1 - 36 * S, top_y + 380 * S
    aw, ah = ax1 - ax0, ay1 - ay0
    x600 = ax0 + int(aw * (600.0 / 3000.0))

    draw.rectangle((ax0, ay0, x600, ay1), fill=(254, 238, 238))
    draw.rectangle((x600, ay0, ax1, ay1), fill=(238, 246, 254))
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gx = ax0 + int(aw * frac)
        draw.line([(gx, ay0), (gx, ay1)], fill=GRID, width=1 * S)
    for y_d in range(ay0, ay1, 18 * S):
        draw.line([(x600, y_d), (x600, min(y_d + 10 * S, ay1))], fill=RED, width=4 * S)

    pts = []
    for i in range(180):
        t = 3000.0 * i / 179.0
        amp = 0.58 * math.exp(-t / 180.0) + 0.42 * math.exp(-t / 820.0)
        pts.append((ax0 + int(aw * (t / 3000.0)), ay1 - int(ah * (0.08 + 0.84 * amp))))
    for i in range(0, 180, 4):
        t = 3000.0 * i / 179.0
        amp = 0.58 * math.exp(-t / 180.0) + 0.42 * math.exp(-t / 820.0)
        v = max(0.02, min(0.97, amp + 0.036 * math.sin(i * 1.9) * math.cos(i * 0.7)))
        px = ax0 + int(aw * (t / 3000.0))
        py = ay1 - int(ah * (0.08 + 0.84 * v))
        r = 3 * S
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(156, 170, 190))
    draw.line(pts, fill=NAVY, width=5 * S)
    draw.line([(ax0, ay0), (ax0, ay1), (ax1, ay1)], fill=NAVY, width=3 * S)

    for tv, lbl in [(0, "0"), (600, "600"), (1800, "1,800"), (3000, "3,000 ms")]:
        gx = ax0 + int(aw * (tv / 3000.0))
        draw.line([(gx, ay1), (gx, ay1 + 8 * S)], fill=NAVY, width=2 * S)
        _centered(draw, gx if tv < 3000 else gx - 20 * S, ay1 + 12 * S, lbl, f_tick, RED if tv == 600 else NAVY)

    draw.text((ax0 - 60 * S, ay0 - 6 * S), "1.0", font=f_tick, fill=NAVY)
    draw.text((ax0 - 60 * S, ay1 - 24 * S), "0.0", font=f_tick, fill=NAVY)
    draw.rounded_rectangle((x600 + 24 * S, ay0 + 14 * S, ax1 - 22 * S, ay0 + 112 * S), radius=10 * S, fill=(255, 255, 255), outline=BLUE, width=2 * S)
    _fit_text(draw, x600 + 36 * S, ax1 - 30 * S, ay0 + 24 * S, "Extended & Trunc. (n = 1,024)", f_lbl, BLUE, "left")
    _fit_text(draw, x600 + 36 * S, ax1 - 30 * S, ay0 + 60 * S, "Valid-window mask: t ≤ t_max", f_sub, NAVY, "left")

    draw.rounded_rectangle((x600 + 24 * S, ay0 + 126 * S, x600 + 346 * S, ay0 + 180 * S), radius=8 * S, fill=(255, 245, 245), outline=RED, width=2 * S)
    _fit_text(draw, x600 + 34 * S, x600 + 336 * S, ay0 + 139 * S, "600 ms cutoff (n = 128)", f_lbl, RED, "center")
    draw.line([(x600 + 24 * S, ay0 + 153 * S), (x600 + 4 * S, ay0 + 153 * S)], fill=RED, width=3 * S)

    # Lower half: Paired Bar Chart on Panel 1
    leg_y = top_y + 444 * S
    draw.rectangle((p1_x0 + 120 * S, leg_y, p1_x0 + 158 * S, leg_y + 26 * S), fill=BLUE, outline=NAVY, width=2 * S)
    draw.text((p1_x0 + 170 * S, leg_y - 2 * S), "A-final", font=f_sec, fill=NAVY)
    draw.rectangle((p1_x0 + 390 * S, leg_y, p1_x0 + 428 * S, leg_y + 26 * S), fill=ORANGE, outline=NAVY, width=2 * S)
    draw.text((p1_x0 + 440 * S, leg_y - 2 * S), "SG+NNLS", font=f_sec, fill=NAVY)

    bx0, by0, bx1, by1 = p1_x0 + 104 * S, top_y + 516 * S, p1_x1 - 36 * S, top_y + 840 * S
    ymin, ymax = 0.84, 0.98
    def y_p1(v): return int(by1 - (by1 - by0) * ((v - ymin) / (ymax - ymin)))

    for tk in [0.85, 0.90, 0.95]:
        yy = y_p1(tk)
        draw.line([(bx0, yy), (bx1, yy)], fill=GRID, width=2 * S)
        draw.text((bx0 - 66 * S, yy - 13 * S), f"{tk:.2f}", font=f_tick, fill=NAVY)
    draw.line([(bx0, by0), (bx0, by1), (bx1, by1)], fill=NAVY, width=3 * S)
    draw.text((p1_x0 + 28 * S, by0 - 34 * S), "Mean r", font=f_lbl, fill=NAVY)

    bw, gap = 82 * S, 16 * S
    g1x = bx0 + int((bx1 - bx0) * 0.27)
    g2x = bx0 + int((bx1 - bx0) * 0.74)

    for cx, va, vb, d_txt, d_col, g_lbl in [
        (g1x, 0.945, 0.916, "Δr = +0.029", BLUE, "1,024 Sims"),
        (g2x, 0.887, 0.929, "Δr = −0.042", ORANGE, "600 ms Stress"),
    ]:
        xA0, xA1 = cx - bw - gap // 2, cx - gap // 2
        xB0, xB1 = cx + gap // 2, cx + gap // 2 + bw
        yA, yB = y_p1(va), y_p1(vb)
        draw.rectangle((xA0, yA, xA1, by1), fill=BLUE, outline=NAVY, width=2 * S)
        draw.rectangle((xB0, yB, xB1, by1), fill=ORANGE, outline=NAVY, width=2 * S)
        _centered(draw, (xA0 + xA1) / 2.0, yA - 36 * S, f"{va:.3f}", f_val, BLUE)
        _centered(draw, (xB0 + xB1) / 2.0, yB - 36 * S, f"{vb:.3f}", f_val, ORANGE)
        top_bar = min(yA, yB)
        _centered(draw, cx, top_bar - 72 * S, d_txt, f_badge, d_col)
        _centered(draw, cx, by1 + 12 * S, g_lbl, f_sec, NAVY)

    _fit_text(draw, p1_x0 + 24 * S, p1_x1 - 24 * S, bot_y - 46 * S,
              "Moderate trunc.: A-final leads | Severe cutoff: NNLS leads", f_lbl, SLATE, "center")

    # =========================================================================
    # PANEL 2: Protocol Reversal Interaction (n = 35 Measured Holdout)
    # =========================================================================
    p2_x0, p2_x1 = panels[1][0], panels[1][1]
    ix0, iy0, ix1, iy1 = p2_x0 + 114 * S, top_y + 128 * S, p2_x1 - 48 * S, top_y + 478 * S
    pymin, pymax = 0.886, 0.910
    def y_p2(v): return int(iy1 - (iy1 - iy0) * ((v - pymin) / (pymax - pymin)))

    # Subtle shaded zones for P-mag vs P-pea
    xm = ix0 + int((ix1 - ix0) * 0.25)
    xp = ix0 + int((ix1 - ix0) * 0.75)
    for tk in [0.890, 0.895, 0.900, 0.905]:
        yy = y_p2(tk)
        draw.line([(ix0, yy), (ix1, yy)], fill=GRID, width=2 * S)
        draw.text((p2_x0 + 26 * S, yy - 13 * S), f"{tk:.3f}", font=f_tick, fill=NAVY)
    draw.line([(ix0, iy0), (ix0, iy1), (ix1, iy1)], fill=NAVY, width=3 * S)
    draw.text((p2_x0 + 26 * S, iy0 - 34 * S), "Mean ref. r", font=f_lbl, fill=NAVY)

    # Slopes connecting P-mag -> P-pea
    y_A_m, y_A_p = y_p2(0.8948), y_p2(0.8929)
    y_S_m, y_S_p = y_p2(0.8909), y_p2(0.9042)
    draw.line([(xm, y_A_m), (xp, y_A_p)], fill=BLUE, width=7 * S)
    draw.line([(xm, y_S_m), (xp, y_S_p)], fill=ORANGE, width=7 * S)

    rd = 13 * S
    for cx, cy, col in [(xm, y_A_m, BLUE), (xp, y_A_p, BLUE), (xm, y_S_m, ORANGE), (xp, y_S_p, ORANGE)]:
        draw.ellipse((cx - rd, cy - rd, cx + rd, cy + rd), fill=col, outline=(255, 255, 255), width=3 * S)

    # Direct high-contrast data labels cleanly positioned away from lines
    draw.text((xm - 108 * S, y_A_m - 42 * S), "A: 0.8948", font=f_val, fill=BLUE)
    draw.text((xm - 116 * S, y_S_m + 18 * S), "SG: 0.8909", font=f_val, fill=ORANGE)
    draw.text((xp - 38 * S, y_S_p - 48 * S), "SG: 0.9042", font=f_val, fill=ORANGE)
    draw.text((xp - 34 * S, y_A_p + 20 * S), "A: 0.8929", font=f_val, fill=BLUE)

    _centered(draw, xm, iy1 + 14 * S, "P-mag (raw |I+iQ|)", f_sec, NAVY)
    _centered(draw, xp, iy1 + 14 * S, "P-pea (vendor peak)", f_sec, NAVY)

    # Two concise Forward RMSE badges right below columns
    card_y0, card_y1 = iy1 + 66 * S, iy1 + 150 * S
    draw.rounded_rectangle((p2_x0 + 36 * S, card_y0, (p2_x0 + p2_x1) // 2 - 12 * S, card_y1),
                           radius=12 * S, fill=BLUE_LT, outline=BLUE, width=2 * S)
    _fit_text(draw, p2_x0 + 44 * S, (p2_x0 + p2_x1) // 2 - 20 * S, card_y0 + 10 * S,
              "P-mag Fwd RMSE", f_lbl, BLUE, "center")
    _fit_text(draw, p2_x0 + 44 * S, (p2_x0 + p2_x1) // 2 - 20 * S, card_y0 + 44 * S,
              "0.0326 vs. 0.0369", f_badge, NAVY, "center")

    draw.rounded_rectangle(((p2_x0 + p2_x1) // 2 + 12 * S, card_y0, p2_x1 - 36 * S, card_y1),
                           radius=12 * S, fill=ORANGE_LT, outline=ORANGE, width=2 * S)
    _fit_text(draw, (p2_x0 + p2_x1) // 2 + 20 * S, p2_x1 - 44 * S, card_y0 + 10 * S,
              "P-pea Fwd RMSE", f_lbl, ORANGE, "center")
    _fit_text(draw, (p2_x0 + p2_x1) // 2 + 20 * S, p2_x1 - 44 * S, card_y0 + 44 * S,
              "0.0097 vs. 0.0163", f_badge, NAVY, "center")

    # Interaction & Common-delta takeaway card
    box2_y0, box2_y1 = card_y1 + 22 * S, bot_y - 24 * S
    draw.rounded_rectangle((p2_x0 + 36 * S, box2_y0, p2_x1 - 36 * S, box2_y1),
                           radius=14 * S, fill=(242, 246, 251), outline=NAVY, width=3 * S)
    _fit_text(draw, p2_x0 + 48 * S, p2_x1 - 48 * S, box2_y0 + 22 * S,
              "Method × Protocol: Δ_int = +0.0152", f_hdr, BLUE, "center")
    _fit_text(draw, p2_x0 + 48 * S, p2_x1 - 48 * S, box2_y0 + 74 * S,
              "95% CI > 0 (record & 7-oil clustered)", f_sec, NAVY, "center")
    _fit_text(draw, p2_x0 + 48 * S, p2_x1 - 48 * S, box2_y0 + 124 * S,
              "Significant across Common-δ (−0.14 to −0.1725)", f_badge, TEAL, "center")
    _fit_text(draw, p2_x0 + 48 * S, p2_x1 - 48 * S, box2_y0 + 174 * S,
              "Fix protocol; report spectral r + Fwd RMSE", f_lbl, SLATE, "center")

    # =========================================================================
    # PANEL 3: Cascade Denoising (top) + Zynq-7020 Execution Budget (bottom)
    # =========================================================================
    p3_x0, p3_x1 = panels[2][0], panels[2][1]
    _fit_text(draw, p3_x0 + 28 * S, p3_x1 - 28 * S, top_y + 88 * S,
              "15-Rec Cascade @ α = 1.0 (NNLS Downstream)", f_sec, NAVY, "center")

    cx0, cx1 = p3_x0 + 206 * S, p3_x1 - 160 * S
    cy_base = top_y + 136 * S
    row_h = 68 * S
    c_min, c_max = 0.85, 0.94
    def x_c(v): return int(cx0 + (cx1 - cx0) * ((v - c_min) / (c_max - c_min)))

    c_rows = [
        ("Raw", 0.885, (140, 150, 165), "Peak err 0.334"),
        ("SG(21,2)", 0.912, BLUE, "Peak err 0.145"),
        ("Edge TCN", 0.919, TEAL, "Peak err 0.149"),
    ]
    for idx, (lbl, val, col, p_txt) in enumerate(c_rows):
        ry0 = cy_base + idx * row_h
        ry1 = ry0 + 46 * S
        x_end = x_c(val)
        _fit_text(draw, p3_x0 + 24 * S, cx0 - 14 * S, ry0 + 6 * S, lbl, f_sec, NAVY, "right")
        draw.rectangle((cx0, ry0, x_end, ry1), fill=col, outline=NAVY, width=2 * S)
        draw.text((x_end + 12 * S, ry0 + 4 * S), f"r = {val:.3f}", font=f_val, fill=col)
    draw.line([(cx0, cy_base - 6 * S), (cx0, cy_base + 3 * row_h - 16 * S)], fill=NAVY, width=3 * S)

    draw.rounded_rectangle((p3_x0 + 32 * S, top_y + 342 * S, p3_x1 - 32 * S, top_y + 428 * S),
                           radius=12 * S, fill=ORANGE_LT, outline=ORANGE, width=2 * S)
    _fit_text(draw, p3_x0 + 42 * S, p3_x1 - 42 * S, top_y + 352 * S,
              "α = 1.0 peak err: 0.334 → 0.145 (SG) / 0.149 (Edge)", f_lbl, NAVY, "center")
    _fit_text(draw, p3_x0 + 42 * S, p3_x1 - 42 * S, top_y + 390 * S,
              "Low/mod. noise (α ≤ 0.3): |Δr| ≤ 0.0006", f_lbl, ORANGE, "center")

    # Divider + ARM Benchmark Section
    draw.line([(p3_x0 + 28 * S, top_y + 448 * S), (p3_x1 - 28 * S, top_y + 448 * S)], fill=BORDER, width=2 * S)
    _fit_text(draw, p3_x0 + 28 * S, p3_x1 - 28 * S, top_y + 464 * S,
              "Zynq-7020 ARM @ 4,000 Echoes (14.25 kB)", f_sec, NAVY, "center")

    draw.rounded_rectangle((p3_x0 + 32 * S, top_y + 510 * S, p3_x1 - 32 * S, top_y + 566 * S),
                           radius=10 * S, fill=TEAL_LT, outline=TEAL, width=2 * S)
    _fit_text(draw, p3_x0 + 42 * S, p3_x1 - 42 * S, top_y + 523 * S,
              "Float32 Parity: r = 1.000000 (rel. RMSE ≤ 3.2e-6)", f_badge, TEAL, "center")

    tx0, tx1 = p3_x0 + 272 * S, p3_x1 - 208 * S
    ty_base = top_y + 596 * S
    t_row_h = 74 * S
    t_bars = [
        ("15-rec Replay (2T)", 16.99, ORANGE, "16.99 s harness"),
        ("Edge Compute (2T)", 14.04, BLUE, "14.04 s"),
        ("SG(21,2) Comp. (1T)", 0.179, TEAL, "0.179 s (78.5×)"),
    ]
    for idx, (lbl, sec_v, col, val_s) in enumerate(t_bars):
        ry0 = ty_base + idx * t_row_h
        ry1 = ry0 + 46 * S
        bar_w = max(8 * S, int((tx1 - tx0) * (sec_v / 17.5)))
        _fit_text(draw, p3_x0 + 18 * S, tx0 - 12 * S, ry0 + 8 * S, lbl, f_lbl, NAVY, "right")
        draw.rectangle((tx0, ry0, tx0 + bar_w, ry1), fill=col, outline=NAVY, width=2 * S)
        _fit_text(draw, tx0 + bar_w + 10 * S, p3_x1 - 18 * S, ry0 + 8 * S, val_s, f_badge, col, "left")
    draw.line([(tx0, ty_base - 6 * S), (tx0, ty_base + 3 * t_row_h - 18 * S)], fill=NAVY, width=3 * S)

    _fit_text(draw, p3_x0 + 24 * S, p3_x1 - 24 * S, bot_y - 46 * S,
              "Deploy neural denoise only if cascade gain warrants latency", f_lbl, SLATE, "center")

    return img.resize((2500, 1000), Image.Resampling.LANCZOS)


def render_ga():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "GA_JMR_20261113.png"
    out_tif = OUT_DIR / "GA_JMR_20261113.tif"
    final_img = build_ga_image()
    final_img.save(str(out_png), format="PNG", dpi=(300, 300), optimize=True)
    final_img.save(str(out_tif), format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    print(f"Saved Graphical Abstract PNG & TIFF: {out_png} ({final_img.size[0]}x{final_img.size[1]}, 300 dpi)")


if __name__ == "__main__":
    render_ga()
