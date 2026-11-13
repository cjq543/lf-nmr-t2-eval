# -*- coding: utf-8 -*-
"""
Regenerate the Graphical Abstract (01_论文正文/figures_EN/GA_JMR_draft_20261106.png)
at 2660 x 1400 px (rendered at 2x = 5320 x 2800 px and downsampled with LANCZOS).
Strictly aligned with Main Text Tables 1, 3, 4, 5, 6 and Sections 3.1-3.5:
- Forward RMSE on 35 benchtop holdout records: A-final is lower under both P-mag (0.0326 vs. 0.0369)
  and P-pea (0.0097 vs. 0.0163).
- Cascade noise ladder (15 records, 3 upstream paths: Raw, SG(21,2), Edge ECA-DS-TCN):
  - At alpha = 1.0 with NNLS downstream: mean r rises from 0.885 (Raw) to 0.912 (SG(21,2)) and 0.919 (Edge);
    dominant-peak error drops from 0.3335 decade (Raw) to 0.1450 (SG(21,2)) and 0.1494 (Edge).
  - With fixed A-original downstream: at alpha = 1.0, mean r is 0.888 (Raw), 0.893 (SG(21,2)), 0.894 (Edge);
    at alpha <= 0.3, un-denoised Raw has the highest observed mean r.
- Zynq-7020 ARM replay (15 records, 3,563 float32 params = 14.25 kB):
  - 4,000-echo record: dual-core Edge ECA-DS-TCN takes 14.04 s vs. 0.179 s for single-core dual-channel SG(21,2).
  - 3,000- / 1,800-echo records (Fig. 6): dual-core Edge takes 10.52 s / 6.32 s.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "01_论文正文" / "figures_EN" / "GA_JMR_draft_20261106.png"
BACKUP_OUT = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106" / "GA_JMR_draft_20261106.png"

FONT_DIR = Path(r"C:\Windows\Fonts")


def load_font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def draw_rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def render_ga():
    S = 2
    W, H = 2660 * S, 1400 * S
    img = Image.new("RGB", (W, H), (249, 250, 252))
    draw = ImageDraw.Draw(img)

    f_title = load_font("segoeuib.ttf", 38 * S)
    f_sub = load_font("segoeuii.ttf", 22 * S)
    f_panel_hdr = load_font("segoeuib.ttf", 23 * S)
    f_sec_bold = load_font("segoeuib.ttf", 19 * S)
    f_body = load_font("segoeui.ttf", 18 * S)
    f_body_bold = load_font("segoeuib.ttf", 18 * S)
    f_small = load_font("segoeui.ttf", 16 * S)
    f_small_bold = load_font("segoeuib.ttf", 16 * S)
    f_small_ital = load_font("segoeuii.ttf", 16 * S)
    f_num_big = load_font("segoeuib.ttf", 23 * S)

    C_NAVY = (22, 43, 76)
    C_SLATE = (75, 88, 108)
    C_BORDER = (206, 214, 224)
    C_PANEL_BG = (255, 255, 255)
    C_BLUE = (31, 119, 180)
    C_BLUE_LIGHT = (232, 243, 252)
    C_ORANGE = (217, 95, 2)
    C_ORANGE_LIGHT = (254, 241, 230)
    C_RED = (196, 48, 43)
    C_TEAL = (24, 128, 112)
    C_GRAY_BG = (242, 245, 249)

    title_text = "Processing choices affect LF-NMR T₂ recovery and execution cost"
    sub_text = (
        "Valid observation windows, envelope protocols, and downstream inverters "
        "govern method selection under constrained budgets"
    )
    tb = draw.textbbox((0, 0), title_text, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) // 2, 24 * S), title_text, font=f_title, fill=C_NAVY)
    sb = draw.textbbox((0, 0), sub_text, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) // 2, 80 * S), sub_text, font=f_sub, fill=C_SLATE)

    top_y = 130 * S
    bot_y = 1364 * S
    pA_x0, pA_x1 = 36 * S, 846 * S
    pB_x0, pB_x1 = 876 * S, 1736 * S
    pC_x0, pC_x1 = 1766 * S, 2624 * S

    for (x0, x1) in [(pA_x0, pA_x1), (pB_x0, pB_x1), (pC_x0, pC_x1)]:
        draw_rounded_rect(draw, (x0, top_y, x1, bot_y), radius=16 * S, fill=C_PANEL_BG, outline=C_BORDER, width=2 * S)
        draw_rounded_rect(draw, (x0, top_y, x1, top_y + 56 * S), radius=16 * S, fill=(238, 243, 249), outline=C_BORDER, width=2 * S)
        draw.rectangle((x0 + 2 * S, top_y + 40 * S, x1 - 2 * S, top_y + 56 * S), fill=(238, 243, 249))
        draw.line([(x0, top_y + 56 * S), (x1, top_y + 56 * S)], fill=C_BORDER, width=2 * S)

    draw.text((pA_x0 + 22 * S, top_y + 12 * S), "A   Observation Window & Protocol Conditions", font=f_panel_hdr, fill=C_NAVY)
    draw.text((pB_x0 + 22 * S, top_y + 12 * S), "B   Tested Synthetic Suites: Mean Spectral r", font=f_panel_hdr, fill=C_NAVY)
    draw.text((pC_x0 + 22 * S, top_y + 12 * S), "C   Key Findings for Processing-Path Selection", font=f_panel_hdr, fill=C_NAVY)

    # =========================================================================
    # PANEL A: Schematic CPMG Decay + Window Regimes + Protocol Split
    # =========================================================================
    ax_x0, ax_y0 = pA_x0 + 78 * S, top_y + 112 * S
    ax_x1, ax_y1 = pA_x1 - 36 * S, top_y + 636 * S
    ax_w, ax_h = ax_x1 - ax_x0, ax_y1 - ax_y0

    x_600 = ax_x0 + int(ax_w * (600.0 / 3000.0))
    draw.rectangle((ax_x0, ax_y0, x_600, ax_y1), fill=(254, 242, 242))
    draw.rectangle((x_600, ax_y0, ax_x1, ax_y1), fill=(240, 247, 253))

    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gx = ax_x0 + int(ax_w * frac)
        draw.line([(gx, ax_y0), (gx, ax_y1)], fill=(224, 230, 238), width=1 * S)
    for frac in [0.25, 0.5, 0.75]:
        gy = ax_y0 + int(ax_h * frac)
        draw.line([(ax_x0, gy), (ax_x1, gy)], fill=(224, 230, 238), width=1 * S)

    for y_dash in range(ax_y0, ax_y1, 18 * S):
        draw.line([(x_600, y_dash), (x_600, min(y_dash + 10 * S, ax_y1))], fill=C_RED, width=3 * S)

    pts_clean = []
    for i in range(200):
        t_ms = 3000.0 * i / 199.0
        amp = 0.58 * math.exp(-t_ms / 180.0) + 0.42 * math.exp(-t_ms / 820.0)
        px = ax_x0 + int(ax_w * (t_ms / 3000.0))
        py = ax_y1 - int(ax_h * (0.06 + 0.88 * amp))
        pts_clean.append((px, py))

    for i in range(0, 200, 3):
        t_ms = 3000.0 * i / 199.0
        amp = 0.58 * math.exp(-t_ms / 180.0) + 0.42 * math.exp(-t_ms / 820.0)
        pseudo_noise = 0.032 * math.sin(i * 1.9) * math.cos(i * 0.7)
        val = max(0.01, min(0.98, amp + pseudo_noise))
        px = ax_x0 + int(ax_w * (t_ms / 3000.0))
        py = ax_y1 - int(ax_h * (0.06 + 0.88 * val))
        r_dot = 3 * S
        draw.ellipse((px - r_dot, py - r_dot, px + r_dot, py + r_dot), fill=(160, 175, 195))

    draw.line(pts_clean, fill=C_NAVY, width=4 * S)
    draw.line([(ax_x0, ax_y0), (ax_x0, ax_y1), (ax_x1, ax_y1)], fill=C_NAVY, width=3 * S)

    for ms_val in [0, 600, 1200, 1800, 2400, 3000]:
        gx = ax_x0 + int(ax_w * (ms_val / 3000.0))
        draw.line([(gx, ax_y1), (gx, ax_y1 + 8 * S)], fill=C_NAVY, width=2 * S)
        lbl = str(ms_val)
        lb = draw.textbbox((0, 0), lbl, font=f_small)
        draw.text((gx - (lb[2] - lb[0]) // 2, ax_y1 + 10 * S), lbl, font=f_small, fill=C_NAVY)

    xlbl = "Echo time t (ms) on 3,000 ms reference grid (schematic)"
    xb = draw.textbbox((0, 0), xlbl, font=f_small_bold)
    draw.text((ax_x0 + (ax_w - (xb[2] - xb[0])) // 2, ax_y1 + 40 * S), xlbl, font=f_small_bold, fill=C_NAVY)

    for y_val, frac in [("1.0", 0.94), ("0.5", 0.50), ("0.0", 0.06)]:
        gy = ax_y1 - int(ax_h * frac)
        draw.line([(ax_x0 - 8 * S, gy), (ax_x0, gy)], fill=C_NAVY, width=2 * S)
        draw.text((ax_x0 - 42 * S, gy - 12 * S), y_val, font=f_small, fill=C_NAVY)

    draw_rounded_rect(draw, (x_600 + 14 * S, ax_y0 + 18 * S, ax_x1 - 14 * S, ax_y0 + 114 * S),
                      radius=10 * S, fill=(255, 255, 255), outline=C_BLUE, width=2 * S)
    draw.text((x_600 + 26 * S, ax_y0 + 24 * S), "Extended suite (1,024 records):", font=f_small_bold, fill=C_BLUE)
    draw.text((x_600 + 26 * S, ax_y0 + 52 * S), "• 725 complete-window records (3,000 ms)", font=f_small, fill=C_NAVY)
    draw.text((x_600 + 26 * S, ax_y0 + 78 * S), "• 299 randomly truncated (400–2,800 ms)", font=f_small, fill=C_NAVY)

    draw_rounded_rect(draw, (x_600 + 14 * S, ax_y0 + 132 * S, ax_x1 - 14 * S, ax_y0 + 204 * S),
                      radius=10 * S, fill=(255, 245, 245), outline=C_RED, width=2 * S)
    draw.text((x_600 + 26 * S, ax_y0 + 138 * S), "Separate 600 ms severe-cutoff stress suite:", font=f_small_bold, fill=C_RED)
    draw.text((x_600 + 26 * S, ax_y0 + 166 * S), "128 paired records clipped at t_max = 600 ms", font=f_small, fill=C_NAVY)

    boxA_y0 = ax_y1 + 86 * S
    boxA_y1 = bot_y - 24 * S
    draw_rounded_rect(draw, (pA_x0 + 24 * S, boxA_y0, pA_x1 - 24 * S, boxA_y1),
                      radius=12 * S, fill=C_GRAY_BG, outline=C_BORDER, width=2 * S)

    draw.text((pA_x0 + 38 * S, boxA_y0 + 16 * S),
              "Unified Evaluation & Processing Factors:", font=f_sec_bold, fill=C_NAVY)

    bullets_A = [
        ("1. Valid-window masking:",
         "Forward loss & residual metrics integrate only within valid echoes (t ≤ t_valid)."),
        ("2. Two measured envelope protocols (35 benchtop holdout records):",
         "P-mag (raw magnitude interpolation) vs. P-pea (vendor echo-peak envelope)."),
        ("3. Downstream inverter pairing (15-record noise ladder):",
         "Three upstream paths (Raw, SG(21,2), Edge) → NNLS (λ = 1000) vs. A-original."),
        ("4. Embedded verification target (Zynq-7020 ARM Cortex-A9):",
         "14.25 kB float32 ECA-DS-TCN (3,563 params) vs. on-board dual-channel SG(21,2)."),
    ]
    cur_y = boxA_y0 + 54 * S
    max_boxA_x = pA_x1 - 32 * S
    for b_title, b_desc in bullets_A:
        draw.text((pA_x0 + 38 * S, cur_y), b_title, font=f_small_bold, fill=C_BLUE)
        bb = draw.textbbox((pA_x0 + 54 * S, cur_y + 24 * S), b_desc, font=f_small)
        assert bb[2] <= max_boxA_x, f"Panel A bullet overflows: {b_desc}"
        draw.text((pA_x0 + 54 * S, cur_y + 24 * S), b_desc, font=f_small, fill=C_NAVY)
        cur_y += 66 * S

    # =========================================================================
    # PANEL B: Grouped Bar Chart (Extended Suite vs. 600 ms Stress Suite)
    # =========================================================================
    ban_x0, ban_y0 = pB_x0 + 36 * S, top_y + 80 * S
    ban_x1, ban_y1 = pB_x1 - 36 * S, top_y + 148 * S
    draw_rounded_rect(draw, (ban_x0, ban_y0, ban_x1, ban_y1),
                      radius=12 * S, fill=(238, 245, 253), outline=C_BLUE, width=2 * S)
    b_txt1 = "Different advantages across tested suites"
    b_txt2 = "Method performance depends on observation window coverage & tail truncation"
    bt1 = draw.textbbox((0, 0), b_txt1, font=f_sec_bold)
    bt2 = draw.textbbox((0, 0), b_txt2, font=f_small_ital)
    draw.text(((pB_x0 + pB_x1 - (bt1[2] - bt1[0])) // 2, ban_y0 + 8 * S), b_txt1, font=f_sec_bold, fill=C_NAVY)
    draw.text(((pB_x0 + pB_x1 - (bt2[2] - bt2[0])) // 2, ban_y0 + 38 * S), b_txt2, font=f_small_ital, fill=C_SLATE)

    leg_y = top_y + 172 * S
    draw.rectangle((pB_x0 + 70 * S, leg_y, pB_x0 + 106 * S, leg_y + 24 * S), fill=C_BLUE, outline=C_NAVY, width=1 * S)
    draw.text((pB_x0 + 118 * S, leg_y - 2 * S), "A-final (expanded learned model)", font=f_small_bold, fill=C_NAVY)

    draw.rectangle((pB_x0 + 455 * S, leg_y, pB_x0 + 491 * S, leg_y + 24 * S), fill=C_ORANGE, outline=C_NAVY, width=1 * S)
    draw.text((pB_x0 + 503 * S, leg_y - 2 * S), "SG+NNLS (classical baseline)", font=f_small_bold, fill=C_NAVY)

    ch_x0, ch_y0 = pB_x0 + 96 * S, top_y + 228 * S
    ch_x1, ch_y1 = pB_x1 - 44 * S, top_y + 812 * S
    ch_w, ch_h = ch_x1 - ch_x0, ch_y1 - ch_y0
    y_min, y_max = 0.85, 0.97

    def val_to_y(v: float) -> int:
        return ch_y1 - int(ch_h * ((v - y_min) / (y_max - y_min)))

    for tick in [0.86, 0.88, 0.90, 0.92, 0.94, 0.96]:
        gy = val_to_y(tick)
        draw.line([(ch_x0, gy), (ch_x1, gy)], fill=(224, 230, 238), width=2 * S)
        draw.line([(ch_x0 - 8 * S, gy), (ch_x0, gy)], fill=C_NAVY, width=2 * S)
        tl = f"{tick:.2f}"
        draw.text((ch_x0 - 54 * S, gy - 12 * S), tl, font=f_small, fill=C_NAVY)

    draw.line([(ch_x0, ch_y0), (ch_x0, ch_y1), (ch_x1, ch_y1)], fill=C_NAVY, width=3 * S)

    g1_cx = ch_x0 + int(ch_w * 0.27)
    bw = 82 * S
    gap = 16 * S

    b1a_x0, b1a_x1 = g1_cx - bw - gap // 2, g1_cx - gap // 2
    b1a_y0 = val_to_y(0.945)
    draw.rectangle((b1a_x0, b1a_y0, b1a_x1, ch_y1), fill=C_BLUE, outline=C_NAVY, width=2 * S)
    v1a = "0.945"
    vb = draw.textbbox((0, 0), v1a, font=f_num_big)
    draw.text(((b1a_x0 + b1a_x1 - (vb[2] - vb[0])) // 2, b1a_y0 - 34 * S), v1a, font=f_num_big, fill=C_BLUE)

    b1b_x0, b1b_x1 = g1_cx + gap // 2, g1_cx + gap // 2 + bw
    b1b_y0 = val_to_y(0.916)
    draw.rectangle((b1b_x0, b1b_y0, b1b_x1, ch_y1), fill=C_ORANGE, outline=C_NAVY, width=2 * S)
    v1b = "0.916"
    vb = draw.textbbox((0, 0), v1b, font=f_num_big)
    draw.text(((b1b_x0 + b1b_x1 - (vb[2] - vb[0])) // 2, b1b_y0 - 34 * S), v1b, font=f_num_big, fill=C_ORANGE)

    dt1 = "Δr = +0.029 (all 1,024)"
    dtb1 = draw.textbbox((0, 0), dt1, font=f_small_bold)
    draw.text((g1_cx - (dtb1[2] - dtb1[0]) // 2, b1a_y0 - 66 * S), dt1, font=f_small_bold, fill=C_BLUE)

    g2_cx = ch_x0 + int(ch_w * 0.75)

    b2a_x0, b2a_x1 = g2_cx - bw - gap // 2, g2_cx - gap // 2
    b2a_y0 = val_to_y(0.887)
    draw.rectangle((b2a_x0, b2a_y0, b2a_x1, ch_y1), fill=C_BLUE, outline=C_NAVY, width=2 * S)
    v2a = "0.887"
    vb = draw.textbbox((0, 0), v2a, font=f_num_big)
    draw.text(((b2a_x0 + b2a_x1 - (vb[2] - vb[0])) // 2, b2a_y0 - 34 * S), v2a, font=f_num_big, fill=C_BLUE)

    b2b_x0, b2b_x1 = g2_cx + gap // 2, g2_cx + gap // 2 + bw
    b2b_y0 = val_to_y(0.929)
    draw.rectangle((b2b_x0, b2b_y0, b2b_x1, ch_y1), fill=C_ORANGE, outline=C_NAVY, width=2 * S)
    v2b = "0.929"
    vb = draw.textbbox((0, 0), v2b, font=f_num_big)
    draw.text(((b2b_x0 + b2b_x1 - (vb[2] - vb[0])) // 2, b2b_y0 - 34 * S), v2b, font=f_num_big, fill=C_ORANGE)

    dt2 = "Δr = −0.042 (600 ms stress)"
    dtb2 = draw.textbbox((0, 0), dt2, font=f_small_bold)
    draw.text((g2_cx - (dtb2[2] - dtb2[0]) // 2, b2b_y0 - 66 * S), dt2, font=f_small_bold, fill=C_ORANGE)

    g1_lines = [
        "Extended suite:",
        "1,024 records, 299 truncated",
        "(SG(51,3)+NNLS baseline)",
    ]
    for idx, line in enumerate(g1_lines):
        fn = f_small_bold if idx < 2 else f_small
        lb = draw.textbbox((0, 0), line, font=fn)
        draw.text((g1_cx - (lb[2] - lb[0]) // 2, ch_y1 + (14 + idx * 24) * S), line, font=fn, fill=C_NAVY)

    g2_lines = [
        "600 ms stress suite:",
        "128 paired records",
        "(SG(31,2)+NNLS baseline)",
    ]
    for idx, line in enumerate(g2_lines):
        fn = f_small_bold if idx < 2 else f_small
        lb = draw.textbbox((0, 0), line, font=fn)
        draw.text((g2_cx - (lb[2] - lb[0]) // 2, ch_y1 + (14 + idx * 24) * S), line, font=fn, fill=C_NAVY)

    subB_y0 = ch_y1 + 102 * S
    subB_y1 = bot_y - 24 * S
    draw_rounded_rect(draw, (pB_x0 + 24 * S, subB_y0, pB_x1 - 24 * S, subB_y1),
                      radius=12 * S, fill=C_GRAY_BG, outline=C_BORDER, width=2 * S)
    draw.text((pB_x0 + 38 * S, subB_y0 + 14 * S),
              "Sub-Suite Breakdown (Table 3 Verified Metrics):", font=f_sec_bold, fill=C_NAVY)

    sub_rows = [
        "• Full-window subset (725 records):  A-final r = 0.953 vs. SG(51,3)+NNLS r = 0.920 (Δr = +0.032)",
        "• Randomly truncated subset (299 records):  A-final r = 0.928 vs. SG(51,3)+NNLS r = 0.906 (Δr = +0.022)",
        "• 600 ms severe-cutoff suite (128 records):  A-final r = 0.887 vs. SG(31,2)+NNLS r = 0.929 (Δr = −0.042)",
        "• Interpretation: Learned prior improves interpolation under moderate random truncation, whereas",
        "  classical NNLS remains more stable when slow T₂ components lose tail observations at 600 ms.",
    ]
    sy = subB_y0 + 52 * S
    max_sub_x = pB_x1 - 36 * S
    for idx, row_txt in enumerate(sub_rows):
        fn = f_small_bold if idx < 3 else f_small_ital
        col = C_NAVY if idx < 3 else C_SLATE
        rb = draw.textbbox((pB_x0 + 38 * S, sy), row_txt, font=fn)
        assert rb[2] <= max_sub_x, f"Sub-row {idx} overflows Panel B box: {rb[2]} > {max_sub_x}"
        draw.text((pB_x0 + 38 * S, sy), row_txt, font=fn, fill=col)
        sy += 40 * S if idx < 3 else 30 * S

    # =========================================================================
    # PANEL C: Three Core Evidence Cards + Decision Takeaway Banner
    # =========================================================================
    c_x0, c_x1 = pC_x0 + 24 * S, pC_x1 - 24 * S
    max_card_x = c_x1 - 16 * S

    cards = [
        (
            "1. Measured Holdout Evaluation Is Protocol-Sensitive (35 Records)",
            C_BLUE,
            C_BLUE_LIGHT,
            [
                ("P-mag (raw magnitude interpolation):", "A-final r = 0.8948 vs. SG(31,2)+NNLS r = 0.8909 (CI spans zero)"),
                ("P-pea (vendor echo-peak envelope):", "A-final r = 0.8929 vs. SG(31,2)+NNLS r = 0.9042 (Δr = −0.0113)"),
                ("Forward decay residual (RMSE):", "A-final lower in P-mag (0.0326 vs. 0.0369) & P-pea (0.0097 vs. 0.0163)"),
                ("Implication:", "Fix envelope protocol & report both spectral r and forward RMSE."),
            ],
        ),
        (
            "2. Upstream Denoising Gains Depend on Noise & Inverter (15 Records)",
            C_ORANGE,
            C_ORANGE_LIGHT,
            [
                ("Strong injected noise (α = 1.0, NNLS λ = 1000):", "Mean r rises from 0.885 (Raw) to 0.912 (SG(21,2)) and 0.919 (Edge)"),
                ("Main-peak error at α = 1.0 (NNLS λ = 1000):", "Drops from 0.3335 decade (Raw) to 0.1450 (SG) and 0.1494 (Edge)"),
                ("Low-to-moderate noise (α ≤ 0.3):", "Small differences across the 3 paths (|Δr| ≤ 0.0006); CIs span zero"),
                ("Fixed A-original inverter:", "α = 1.0 r is 0.888 (Raw), 0.893 (SG), 0.894 (Edge); Raw highest at α ≤ 0.3"),
            ],
        ),
        (
            "3. Zynq-7020 ARM Replay Consistency & Compute Benchmarks",
            C_TEAL,
            (232, 248, 245),
            [
                ("15-record file replay (14.25 kB, 3,563 params):", "Matches host float32 (r = 1.000000, RMSE ≤ 3.2×10⁻⁶; 4k-echo harness ~16.99 s)"),
                ("Separate 4,000-echo compute benchmark:", "Dual-core Edge ECA-DS-TCN: 14.04 s vs. 1-core 2-ch SG(21,2): 0.179 s"),
                ("3,000- / 1,800-echo compute benchmark:", "Dual-core: 10.52 s / 6.32 s (single-core: 19.57 s / 11.78 s)"),
                ("Engineering rule:", "Only consider neural denoise when cascade gain justifies its latency."),
            ],
        ),
    ]

    cy = top_y + 78 * S
    card_h = 318 * S
    for title_c, border_c, hdr_bg, rows_c in cards:
        draw_rounded_rect(draw, (c_x0, cy, c_x1, cy + card_h),
                          radius=12 * S, fill=(255, 255, 255), outline=border_c, width=2 * S)
        draw_rounded_rect(draw, (c_x0, cy, c_x1, cy + 46 * S),
                          radius=12 * S, fill=hdr_bg, outline=border_c, width=2 * S)
        draw.rectangle((c_x0 + 2 * S, cy + 32 * S, c_x1 - 2 * S, cy + 46 * S), fill=hdr_bg)
        draw.line([(c_x0, cy + 46 * S), (c_x1, cy + 46 * S)], fill=border_c, width=2 * S)
        draw.text((c_x0 + 18 * S, cy + 10 * S), title_c, font=f_sec_bold, fill=C_NAVY)

        ry = cy + 58 * S
        for k_txt, v_txt in rows_c:
            kb = draw.textbbox((c_x0 + 20 * S, ry), "• " + k_txt, font=f_small_bold)
            vb = draw.textbbox((c_x0 + 38 * S, ry + 24 * S), v_txt, font=f_small)
            assert kb[2] <= max_card_x, f"Card key overflows: {k_txt}"
            assert vb[2] <= max_card_x, f"Card value overflows: {v_txt} ({vb[2]} > {max_card_x})"
            draw.text((c_x0 + 20 * S, ry), "• " + k_txt, font=f_small_bold, fill=C_NAVY)
            draw.text((c_x0 + 38 * S, ry + 24 * S), v_txt, font=f_small, fill=C_SLATE)
            ry += 62 * S

        cy += card_h + 22 * S

    syn_y0 = cy + 2 * S
    syn_y1 = bot_y - 24 * S
    draw_rounded_rect(draw, (c_x0, syn_y0, c_x1, syn_y1),
                      radius=12 * S, fill=C_NAVY, outline=C_NAVY, width=2 * S)
    s_line1 = "Decision Basis for Constrained LF-NMR Processing (Table 6):"
    s_line2 = "Select paths jointly by valid window, fixed envelope protocol,"
    s_line3 = "downstream inverter response, and measured preprocessing cost."
    draw.text((c_x0 + 22 * S, syn_y0 + 14 * S), s_line1, font=f_sec_bold, fill=(255, 218, 128))
    draw.text((c_x0 + 22 * S, syn_y0 + 50 * S), s_line2, font=f_body_bold, fill=(255, 255, 255))
    draw.text((c_x0 + 22 * S, syn_y0 + 82 * S), s_line3, font=f_body_bold, fill=(255, 255, 255))

    final_img = img.resize((2660, 1400), Image.Resampling.LANCZOS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(str(OUT_PATH), format="PNG", optimize=True)
    final_img.save(str(BACKUP_OUT), format="PNG", optimize=True)
    print(f"Saved Graphical Abstract: {OUT_PATH} ({final_img.size[0]}x{final_img.size[1]})")


if __name__ == "__main__":
    render_ga()
