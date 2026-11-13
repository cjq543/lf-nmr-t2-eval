# -*- coding: utf-8 -*-
"""
Regenerate clean, publication-grade Figure 3 (forest plot of paired effect sizes Δr)
for both English (fig_EN_effect_forest.png) and Chinese (fig_CN_effect_forest.png) JMR v2 manuscripts
using high-resolution 2x supersampled TrueType rendering (Pillow + LANCZOS antialiasing).

Fixes P1-3 from 中英文v2同步独立审阅结论_20261106.md:
1. Eliminates text-on-errorbar collisions and right-border overflow by placing all
   numerical annotations (Δr [95% CI]) in a dedicated right-hand tabular column
   inside each panel, separated from the plot area by a subtle divider.
2. Removes cluttered "significant", "crosses zero (n.s.)", and "(win/loss: ...)" tags
   from Panel (a).
3. Shortens y-axis labels and enlarges typography so all text remains crisp and
   effortlessly readable when scaled to Word's 148.17 mm text width.
4. Preserves 100% of the exact frozen numbers from 01_论文正文/effect_forest_numbers.csv.
"""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "01_论文正文" / "effect_forest_numbers.csv"
OUT_DIR = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106" / "fig3_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with CSV_PATH.open("r", encoding="utf-8-sig") as f:
        return {(r["panel"], r["comparison"]): r for r in csv.DictReader(f)}


def fmt_signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def get_fonts(lang: str, scale: int = 2):
    font_dir = Path(r"C:\Windows\Fonts")
    if lang == "EN":
        reg = str(font_dir / "arial.ttf")
        bold = str(font_dir / "arialbd.ttf")
    else:
        reg = str(font_dir / "msyh.ttc")
        bold = str(font_dir / "msyhbd.ttc")
    return {
        "title": ImageFont.truetype(bold, 32 * scale),
        "axis_title": ImageFont.truetype(bold, 31 * scale),
        "col_hdr": ImageFont.truetype(bold, 30 * scale),
        "label": ImageFont.truetype(reg, 30 * scale),
        "tick": ImageFont.truetype(reg, 27 * scale),
        "val_bold": ImageFont.truetype(bold, 29 * scale),
        "val_reg": ImageFont.truetype(reg, 29 * scale),
        "legend": ImageFont.truetype(reg, 27 * scale),
    }


def draw_dashed_hline(draw, x0, x1, y, color, width, dash=14, gap=9):
    x = x0
    while x < x1:
        xe = min(x + dash, x1)
        draw.line([(x, y), (xe, y)], fill=color, width=width)
        x += dash + gap


def draw_dashed_vline(draw, x, y0, y1, color, width, dash=14, gap=9):
    y = y0
    while y < y1:
        ye = min(y + dash, y1)
        draw.line([(x, y), (x, ye)], fill=color, width=width)
        y += dash + gap


def render_forest(lang: str, out_path: Path, target_w: int = 2710, target_h: int = 1065):
    r_map = load_rows()
    S = 2  # 2x supersampling
    W, H = target_w * S, target_h * S
    im = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    F = get_fonts(lang, scale=S)

    if lang == "EN":
        p1_keys = [
            ("SG(31,2)+NNLS (λ=50)", "SG旧_r"),
            ("SG(51,3)+NNLS (λ=200)", "SG扩展_r"),
            ("A-original (T06)", "A旧版_r"),
            ("Mask-fair NNLS (λ=1000)", "修正NNLS_r"),
        ]
        p2_keys = [
            ("V6: 600 ms window", "V6_缺失时间窗_600ms"),
            ("V5: Strong baseline drift", "V5_强基线漂移"),
            ("V3: Fast-end extrap.", "V3_峰位外推_快端"),
            ("V6: 1200 ms window", "V6_缺失时间窗_1200ms"),
            ("V3: Slow-end extrap.", "V3_峰位外推_慢端"),
            ("V2: Broader peaks", "V2_更宽峰宽"),
            ("V4: 4–5 components", "V4_多组分4-5峰"),
            ("V7: Low SNR (5–8 dB)", "V7_极低幅比5-8dB"),
            ("V1: Narrower peaks", "V1_更窄峰宽"),
        ]
        t_a = "(a) 35 Held-Out Records (P-mag): A-final vs. Baselines"
        t_b = "(b) 9 Stress Conditions (128/cond.): A-final vs. SG(31,2)+NNLS"
        xl_a = "Paired Δr (A-final minus comparator)"
        xl_b = "Paired Δr (A-final minus SG+NNLS)"
        leg_rec = "Record-level 95% CI (n=35)"
        leg_clu = "Clustered 95% CI (7 oil types)"
        col_hdr = "Δr [95% CI]"
    else:
        p1_keys = [
            ("SG(31,2)+NNLS (λ=50)", "SG旧_r"),
            ("SG(51,3)+NNLS (λ=200)", "SG扩展_r"),
            ("A-original (T06)", "A旧版_r"),
            ("掩码修正 NNLS (λ=1000)", "修正NNLS_r"),
        ]
        p2_keys = [
            ("V6: 600 ms 严重截断", "V6_缺失时间窗_600ms"),
            ("V5: 强基线漂移", "V5_强基线漂移"),
            ("V3: 快端峰位外推", "V3_峰位外推_快端"),
            ("V6: 1200 ms 轻度截断", "V6_缺失时间窗_1200ms"),
            ("V3: 慢端峰位外推", "V3_峰位外推_慢端"),
            ("V2: 更宽峰宽", "V2_更宽峰宽"),
            ("V4: 4–5 峰多组分", "V4_多组分4-5峰"),
            ("V7: 极低信噪比 (5–8 dB)", "V7_极低幅比5-8dB"),
            ("V1: 更窄峰宽", "V1_更窄峰宽"),
        ]
        t_a = "(a) 35 条历史留出实测 (P-mag)：A-final 对比各基线"
        t_b = "(b) 9 项受控扰动 (各 128 条)：A-final 对比 SG(31,2)+NNLS"
        xl_a = "配对相关系数差值 Δr（A-final 减对比方法）"
        xl_b = "配对相关系数差值 Δr（A-final 减 SG+NNLS）"
        leg_rec = "记录级 95% CI (n=35)"
        leg_clu = "油种聚类 95% CI (7 组)"
        col_hdr = "Δr [95% CI]"

    # Vertical geometry (in 1x coordinates, multiplied by S)
    top_y = 104 * S
    bot_y = 1085 * S

    # -------------------------------------------------------------------------
    # Panel (a) horizontal geometry (1x coordinates)
    # -------------------------------------------------------------------------
    a_plot_x0 = 372 * S
    a_plot_x1 = 792 * S
    a_col_x0 = 806 * S
    a_box_x1 = 1184 * S
    x1_min, x1_max = -0.015, 0.052

    def x_map_a(v: float) -> float:
        return a_plot_x0 + (v - x1_min) / (x1_max - x1_min) * (a_plot_x1 - a_plot_x0)

    # Panel (a) title & header
    draw.text((20 * S, 18 * S), t_a, font=F["title"], fill=(17, 24, 39))
    draw.text((a_col_x0, top_y - 42 * S), col_hdr, font=F["col_hdr"], fill=(51, 65, 85))

    # Panel (a) grid & axes box
    for xv, xlbl in [(-0.01, None), (0.00, "0.00"), (0.01, None), (0.02, "+0.02"), (0.03, None), (0.04, "+0.04"), (0.05, None)]:
        xp = x_map_a(xv)
        if xv == 0.0:
            draw_dashed_vline(draw, xp, top_y, bot_y, (90, 100, 115), 3 * S, dash=14 * S, gap=9 * S)
        else:
            draw_dashed_vline(draw, xp, top_y, bot_y, (215, 222, 232), 2 * S, dash=6 * S, gap=8 * S)
        draw.line([(xp, bot_y), (xp, bot_y + (9 if xlbl else 5) * S)], fill=(60, 60, 60), width=2 * S)
        if xlbl:
            bb = draw.textbbox((0, 0), xlbl, font=F["tick"])
            tw = bb[2] - bb[0]
            draw.text((xp - tw / 2, bot_y + 13 * S), xlbl, font=F["tick"], fill=(30, 41, 59))

    # Subtle shaded band for right-hand value column
    draw.rectangle([a_plot_x1, top_y, a_box_x1, bot_y], fill=(248, 250, 252))
    draw.line([(a_plot_x1, top_y), (a_plot_x1, bot_y)], fill=(203, 213, 225), width=2 * S)
    draw.rectangle([a_plot_x0, top_y, a_box_x1, bot_y], outline=(71, 85, 105), width=2 * S)

    # Panel (a) rows (4 items from bottom to top, leaving bottom room for legend)
    y1_positions = [790 * S, 595 * S, 400 * S, 205 * S]
    for (label, key), yp in zip(p1_keys, y1_positions):
        row = r_map[("holdout35", key)]
        m = float(row["dmean"])
        lo = float(row["ci95_lo"])
        hi = float(row["ci95_hi"])
        clo = float(row["cluster_lo"])
        chi = float(row["cluster_hi"])
        sig = lo > 0 or hi < 0
        c_main = (28, 93, 153) if sig else (105, 115, 125)
        c_clu = (59, 130, 246)

        draw.line([(a_plot_x0, yp), (a_box_x1, yp)], fill=(241, 245, 249), width=1 * S)
        draw.line([(a_plot_x0 - 8 * S, yp), (a_plot_x0, yp)], fill=(60, 60, 60), width=2 * S)
        bb = draw.textbbox((0, 0), label, font=F["label"])
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((a_plot_x0 - 14 * S - tw, yp - th / 2 - 2 * S), label, font=F["label"], fill=(17, 24, 39))

        # Record-level 95% CI (upper bar)
        y_rec = yp - 16 * S
        x_lo, x_hi, x_m = x_map_a(lo), x_map_a(hi), x_map_a(m)
        draw.line([(x_lo, y_rec), (x_hi, y_rec)], fill=c_main, width=4 * S)
        draw.line([(x_lo, y_rec - 10 * S), (x_lo, y_rec + 10 * S)], fill=c_main, width=3 * S)
        draw.line([(x_hi, y_rec - 10 * S), (x_hi, y_rec + 10 * S)], fill=c_main, width=3 * S)
        r_pt = 8 * S
        draw.ellipse([x_m - r_pt, y_rec - r_pt, x_m + r_pt, y_rec + r_pt], fill=c_main)

        # Clustered 95% CI (lower dashed bar)
        y_clu = yp + 18 * S
        x_clo, x_chi = x_map_a(clo), x_map_a(chi)
        draw_dashed_hline(draw, x_clo, x_chi, y_clu, c_clu, 3 * S, dash=12 * S, gap=7 * S)
        draw.line([(x_clo, y_clu - 9 * S), (x_clo, y_clu + 9 * S)], fill=c_clu, width=3 * S)
        draw.line([(x_chi, y_clu - 9 * S), (x_chi, y_clu + 9 * S)], fill=c_clu, width=3 * S)

        # Right-hand column value
        val_txt = f"{fmt_signed(m)} [{fmt_signed(lo)}, {fmt_signed(hi)}]"
        f_val = F["val_bold"] if sig else F["val_reg"]
        c_val = (17, 24, 39) if sig else (71, 85, 105)
        vbb = draw.textbbox((0, 0), val_txt, font=f_val)
        vth = vbb[3] - vbb[1]
        draw.text((a_col_x0, yp - vth / 2 - 2 * S), val_txt, font=f_val, fill=c_val)

    # Legend box inside Panel (a) bottom-left
    leg_x0, leg_y0 = a_plot_x0 + 10 * S, 912 * S
    leg_x1, leg_y1 = a_plot_x1 - 10 * S, 1062 * S
    draw.rectangle([leg_x0, leg_y0, leg_x1, leg_y1], fill=(255, 255, 255), outline=(203, 213, 225), width=2 * S)
    ly1 = leg_y0 + 40 * S
    draw.line([(leg_x0 + 14 * S, ly1), (leg_x0 + 66 * S, ly1)], fill=(28, 93, 153), width=4 * S)
    draw.ellipse([leg_x0 + 40 * S - 7 * S, ly1 - 7 * S, leg_x0 + 40 * S + 7 * S, ly1 + 7 * S], fill=(28, 93, 153))
    draw.text((leg_x0 + 78 * S, ly1 - 16 * S), leg_rec, font=F["legend"], fill=(30, 41, 59))
    ly2 = leg_y0 + 104 * S
    draw_dashed_hline(draw, leg_x0 + 14 * S, leg_x0 + 66 * S, ly2, (59, 130, 246), 3 * S, dash=12 * S, gap=7 * S)
    draw.text((leg_x0 + 78 * S, ly2 - 16 * S), leg_clu, font=F["legend"], fill=(30, 41, 59))

    # Panel (a) X-axis title
    abb = draw.textbbox((0, 0), xl_a, font=F["axis_title"])
    atw = abb[2] - abb[0]
    draw.text(((a_plot_x0 + a_plot_x1) / 2 - atw / 2, bot_y + 58 * S), xl_a, font=F["axis_title"], fill=(17, 24, 39))

    # -------------------------------------------------------------------------
    # Panel (b) horizontal geometry (1x coordinates)
    # -------------------------------------------------------------------------
    b_plot_x0 = 1554 * S
    b_plot_x1 = 1974 * S
    b_col_x0 = 1988 * S
    b_box_x1 = 2366 * S
    x2_min, x2_max = -0.075, 0.155

    def x_map_b(v: float) -> float:
        return b_plot_x0 + (v - x2_min) / (x2_max - x2_min) * (b_plot_x1 - b_plot_x0)

    # Panel (b) title & header
    draw.text((1208 * S, 18 * S), t_b, font=F["title"], fill=(17, 24, 39))
    draw.text((b_col_x0, top_y - 42 * S), col_hdr, font=F["col_hdr"], fill=(51, 65, 85))

    # Panel (b) grid & axes box
    for xv, xlbl in [(-0.05, "−0.05"), (0.00, "0.00"), (0.05, "+0.05"), (0.10, "+0.10"), (0.15, "+0.15")]:
        xp = x_map_b(xv)
        if xv == 0.0:
            draw_dashed_vline(draw, xp, top_y, bot_y, (90, 100, 115), 3 * S, dash=14 * S, gap=9 * S)
        else:
            draw_dashed_vline(draw, xp, top_y, bot_y, (215, 222, 232), 2 * S, dash=6 * S, gap=8 * S)
        draw.line([(xp, bot_y), (xp, bot_y + 9 * S)], fill=(60, 60, 60), width=2 * S)
        bb = draw.textbbox((0, 0), xlbl, font=F["tick"])
        tw = bb[2] - bb[0]
        draw.text((xp - tw / 2, bot_y + 13 * S), xlbl, font=F["tick"], fill=(30, 41, 59))

    draw.rectangle([b_plot_x1, top_y, b_box_x1, bot_y], fill=(248, 250, 252))
    draw.line([(b_plot_x1, top_y), (b_plot_x1, bot_y)], fill=(203, 213, 225), width=2 * S)
    draw.rectangle([b_plot_x0, top_y, b_box_x1, bot_y], outline=(71, 85, 105), width=2 * S)

    # Panel (b) rows (9 items from bottom to top)
    n2 = len(p2_keys)
    y2_step = (bot_y - top_y - 106 * S) / (n2 - 1)
    for idx, (label, key) in enumerate(p2_keys):
        yp = bot_y - 53 * S - idx * y2_step
        row = r_map[("ood9dim_vs_SG31", key)]
        m = float(row["dmean"])
        lo = float(row["ci95_lo"])
        hi = float(row["ci95_hi"])
        if lo > 0:
            c = (21, 115, 71)   # green: A-final higher
            sig = True
        elif hi < 0:
            c = (201, 42, 42)   # red: SG+NNLS higher
            sig = True
        else:
            c = (105, 115, 125) # gray: unresolved
            sig = False

        draw.line([(b_plot_x0, yp), (b_box_x1, yp)], fill=(241, 245, 249), width=1 * S)
        draw.line([(b_plot_x0 - 8 * S, yp), (b_plot_x0, yp)], fill=(60, 60, 60), width=2 * S)
        bb = draw.textbbox((0, 0), label, font=F["label"])
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((b_plot_x0 - 14 * S - tw, yp - th / 2 - 2 * S), label, font=F["label"], fill=(17, 24, 39))

        x_lo, x_hi, x_m = x_map_b(lo), x_map_b(hi), x_map_b(m)
        draw.line([(x_lo, yp), (x_hi, yp)], fill=c, width=4 * S)
        draw.line([(x_lo, yp - 9 * S), (x_lo, yp + 9 * S)], fill=c, width=3 * S)
        draw.line([(x_hi, yp - 9 * S), (x_hi, yp + 9 * S)], fill=c, width=3 * S)
        r_pt = 8 * S
        draw.ellipse([x_m - r_pt, yp - r_pt, x_m + r_pt, yp + r_pt], fill=c)

        val_txt = f"{fmt_signed(m)} [{fmt_signed(lo)}, {fmt_signed(hi)}]"
        f_val = F["val_bold"] if sig else F["val_reg"]
        c_val = (17, 24, 39) if sig else (71, 85, 105)
        vbb = draw.textbbox((0, 0), val_txt, font=f_val)
        vth = vbb[3] - vbb[1]
        draw.text((b_col_x0, yp - vth / 2 - 2 * S), val_txt, font=f_val, fill=c_val)

    bbb = draw.textbbox((0, 0), xl_b, font=F["axis_title"])
    btw = bbb[2] - bbb[0]
    draw.text(((b_plot_x0 + b_plot_x1) / 2 - btw / 2, bot_y + 58 * S), xl_b, font=F["axis_title"], fill=(17, 24, 39))

    out_im = im.resize((target_w, target_h), Image.LANCZOS)
    out_im.save(str(out_path), format="PNG", optimize=True)
    print(f"Saved {lang} Figure 3 -> {out_path} ({out_im.size})")


def main():
    render_forest("EN", OUT_DIR / "fig_EN_effect_forest.png", target_w=2380, target_h=1210)
    render_forest("CN", OUT_DIR / "fig_CN_effect_forest.png", target_w=2380, target_h=1210)


if __name__ == "__main__":
    main()
