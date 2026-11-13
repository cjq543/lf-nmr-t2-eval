# -*- coding: utf-8 -*-
"""
Regenerate Supplementary Figures Fig. S1 (fig_EN_ood_9dim.png / fig_ood_9dim.png)
and Fig. S2 (fig_EN_cnn_compare.png / fig_cnn_compare.png) using explicit key-bound
structured lookups from the authoritative CSVs:
- 02_冻结结果权重与代码/results/ood_rows_corrected.csv (3,456 rows: 9 conditions x 128 records x 3 Table 4 branches)
- 02_冻结结果权重与代码/results/ood_rows.csv (historical pre-fix A-original diagnostic branch)
- 04_审阅核验链/11_基线修正_20261025/corrected_ext_dist_rows.csv (extended-distribution suite, n=1,024: 725 complete, 299 truncated)
- 04_审阅核验链/13_现代基线对照_20261025/smallcnn_test_rows.csv (SmallCNN1D seed 1 on the same n=1,024 extended suite)
No positional array slicing is used anywhere.
"""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OOD_CORR_CSV = ROOT / "02_冻结结果权重与代码" / "results" / "ood_rows_corrected.csv"
OOD_OLD_CSV = ROOT / "02_冻结结果权重与代码" / "results" / "ood_rows.csv"
EXT_CORR_CSV = ROOT / "04_审阅核验链" / "11_基线修正_20261025" / "corrected_ext_dist_rows.csv"
SMALLCNN_CSV = ROOT / "04_审阅核验链" / "13_现代基线对照_20261025" / "smallcnn_test_rows.csv"

FIG_EN_DIR = ROOT / "01_论文正文" / "figures_EN"
FIG_CN_DIR = ROOT / "01_论文正文" / "figures"
AUDIT_DIR = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106"
FONT_DIR = Path(r"C:\Windows\Fonts")


def load_font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def compute_s1_structured_data():
    df_corr = pd.read_csv(OOD_CORR_CSV)
    df_old = pd.read_csv(OOD_OLD_CSV)

    variants = [
        ("V1_更窄峰宽", "V1\nNarrower\npeaks", "V1\n更窄峰宽"),
        ("V2_更宽峰宽", "V2\nBroader\npeaks", "V2\n更宽峰宽"),
        ("V3_峰位外推_快端", "V3-fast\nFast-end\nextrap.", "V3-快端\n快端外推"),
        ("V3_峰位外推_慢端", "V3-slow\nSlow-end\nextrap.", "V3-慢端\n慢端外推"),
        ("V4_多组分4-5峰", "V4\nMulti-comp.\n(4–5 peaks)", "V4\n多组分(4–5峰)"),
        ("V5_强基线漂移", "V5\nStrong\nbaseline drift", "V5\n强基线漂移"),
        ("V6_缺失时间窗_1200ms", "V6-mild\n1,200 ms\ntruncation", "V6-轻度\n1,200 ms截断"),
        ("V6_缺失时间窗_600ms", "V6-severe\n600 ms\ntruncation", "V6-严重\n600 ms截断"),
        ("V7_极低幅比5-8dB", "V7\nLow SNR\n(5–8 dB)", "V7\n极低信噪比"),
    ]

    records = {}
    for v_key, _, _ in variants:
        records[v_key] = {}
        # Historical A-original (T06, pre-fix diagnostic from ood_rows.csv)
        s_old = df_old[(df_old["variant"] == v_key) & (df_old["branch"] == "A仿真版")]["pearson_r"]
        assert len(s_old) == 128, f"Expected 128 rows for A-original in {v_key}, got {len(s_old)}"
        records[v_key]["A_original_hist"] = {
            "mean": float(s_old.mean()),
            "q25": float(s_old.quantile(0.25)),
            "q75": float(s_old.quantile(0.75)),
        }
        # Mask-fair Table 4 branches from ood_rows_corrected.csv
        for b_csv, b_id in [
            ("A仿真版", "A_final"),
            ("SG(31,2)+NNLS", "SG31_NNLS"),
            ("直通NNLS", "Direct_NNLS"),
        ]:
            s_cur = df_corr[(df_corr["variant"] == v_key) & (df_corr["branch"] == b_csv)]["pearson_r"]
            assert len(s_cur) == 128, f"Expected 128 rows for {b_id} in {v_key}, got {len(s_cur)}"
            records[v_key][b_id] = {
                "mean": float(s_cur.mean()),
                "q25": float(s_cur.quantile(0.25)),
                "q75": float(s_cur.quantile(0.75)),
            }

    # Explicit Spot-Check Assertions against Table 4
    assert round(records["V1_更窄峰宽"]["A_final"]["mean"], 3) == 0.918
    assert round(records["V1_更窄峰宽"]["SG31_NNLS"]["mean"], 3) == 0.790
    assert round(records["V1_更窄峰宽"]["Direct_NNLS"]["mean"], 3) == 0.644
    assert round(records["V6_缺失时间窗_600ms"]["A_final"]["mean"], 3) == 0.887
    assert round(records["V6_缺失时间窗_600ms"]["SG31_NNLS"]["mean"], 3) == 0.929
    assert round(records["V6_缺失时间窗_600ms"]["Direct_NNLS"]["mean"], 3) == 0.831
    return variants, records


def render_fig_s1(lang: str = "EN"):
    variants, records = compute_s1_structured_data()
    S = 2
    W, H = 2400 * S, 1120 * S
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if lang == "EN":
        f_title = load_font("segoeuib.ttf", 26 * S)
        f_sub = load_font("segoeuii.ttf", 18 * S)
        f_axis = load_font("segoeuib.ttf", 19 * S)
        f_tick = load_font("segoeui.ttf", 16 * S)
        f_val = load_font("segoeuib.ttf", 13 * S)
        f_leg = load_font("segoeuib.ttf", 16 * S)
    else:
        f_title = load_font("msyhbd.ttc", 25 * S)
        f_sub = load_font("msyh.ttc", 17 * S)
        f_axis = load_font("msyhbd.ttc", 18 * S)
        f_tick = load_font("msyh.ttc", 15 * S)
        f_val = load_font("segoeuib.ttf", 13 * S)
        f_leg = load_font("msyhbd.ttc", 15 * S)

    C_NAVY = (22, 43, 76)
    C_GRID = (224, 230, 238)

    if lang == "EN":
        title = "Nine-Condition Perturbation Stress Suite (Mask-Fair Protocol; 128 Paired Records/Condition)"
        subtitle = "Bars show per-condition mean spectral Pearson r for the three mask-fair Table 4 branches; whiskers indicate 25th–75th IQR"
        ylabel = "Spectral Pearson correlation r"
        branches = [
            ("A_final", (214, 39, 40), "A-final (seed 3, Table 4 primary)"),
            ("SG31_NNLS", (31, 119, 180), "Mask-fair SG(31,2)+NNLS (λ = 50, Table 4)"),
            ("Direct_NNLS", (148, 103, 189), "Mask-fair Direct NNLS (λ = 1000, Table 4)"),
        ]
    else:
        title = "九项扰动压力测试套件逐条件均值与四分位距对比（掩码公平同口径，每条件 128 对记录）"
        subtitle = "柱高表示三项表 4 掩码公平分支的平均谱相关系数 r；竖向误差线表示 25%–75% 分位距（IQR）"
        ylabel = "谱 Pearson 相关系数 r"
        branches = [
            ("A_final", (214, 39, 40), "A-final（seed 3，表 4 主模型）"),
            ("SG31_NNLS", (31, 119, 180), "掩码修正 SG(31,2)+NNLS（λ = 50，表 4）"),
            ("Direct_NNLS", (148, 103, 189), "掩码修正直通 NNLS（λ = 1000，表 4）"),
        ]

    tb = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) // 2, 22 * S), title, font=f_title, fill=C_NAVY)
    sb = draw.textbbox((0, 0), subtitle, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) // 2, 62 * S), subtitle, font=f_sub, fill=(80, 92, 110))

    # Centered legend box at top
    leg_y = 104 * S
    leg_widths = [38 * S + (draw.textbbox((0, 0), lbl, font=f_leg)[2] - draw.textbbox((0, 0), lbl, font=f_leg)[0]) for _, _, lbl in branches]
    leg_gap = 64 * S
    total_leg_w = sum(leg_widths) + leg_gap * (len(branches) - 1)
    lx = (W - total_leg_w) // 2
    for idx_b, (b_id, col, b_label) in enumerate(branches):
        draw.rectangle((lx, leg_y, lx + 30 * S, leg_y + 20 * S), fill=col, outline=C_NAVY, width=1 * S)
        draw.text((lx + 38 * S, leg_y - 2 * S), b_label, font=f_leg, fill=C_NAVY)
        lx += leg_widths[idx_b] + leg_gap

    # Plot area
    px0, py0 = 130 * S, 155 * S
    px1, py1 = W - 48 * S, H - 145 * S
    pw, ph = px1 - px0, py1 - py0
    y_min, y_max = 0.0, 1.12

    def val_to_y(v: float) -> int:
        v_clip = max(y_min, min(y_max, v))
        return py1 - int(ph * ((v_clip - y_min) / (y_max - y_min)))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = val_to_y(tick)
        draw.line([(px0, gy), (px1, gy)], fill=C_GRID, width=2 * S)
        draw.line([(px0 - 8 * S, gy), (px0, gy)], fill=C_NAVY, width=2 * S)
        tl = f"{tick:.1f}"
        draw.text((px0 - 50 * S, gy - 11 * S), tl, font=f_tick, fill=C_NAVY)

    draw.line([(px0, py0), (px0, py1), (px1, py1)], fill=C_NAVY, width=3 * S)
    draw.text((24 * S, py0 - 34 * S), ylabel, font=f_axis, fill=C_NAVY)

    n_groups = len(variants)
    group_step = pw / n_groups
    bw = 46 * S
    offsets = [-1.0 * bw, 0.0, 1.0 * bw]

    for g_idx, (v_key, en_lbl, cn_lbl) in enumerate(variants):
        cx = px0 + int((g_idx + 0.5) * group_step)
        for b_idx, (b_id, col, _) in enumerate(branches):
            stat = records[v_key][b_id]
            mean_v = stat["mean"]
            q25_v = stat["q25"]
            q75_v = stat["q75"]

            bx0 = int(cx + offsets[b_idx] - bw * 0.44)
            bx1 = int(cx + offsets[b_idx] + bw * 0.44)
            by0 = val_to_y(mean_v)
            draw.rectangle((bx0, by0, bx1, py1), fill=col, outline=C_NAVY, width=1 * S)

            # Draw IQR whisker (q25 to q75)
            wx = (bx0 + bx1) // 2
            wy_lo = val_to_y(q25_v)
            wy_hi = val_to_y(q75_v)
            draw.line([(wx, wy_hi), (wx, wy_lo)], fill=(30, 30, 30), width=2 * S)
            cap = 6 * S
            draw.line([(wx - cap, wy_hi), (wx + cap, wy_hi)], fill=(30, 30, 30), width=2 * S)
            draw.line([(wx - cap, wy_lo), (wx + cap, wy_lo)], fill=(30, 30, 30), width=2 * S)

            # Value label above the higher of mean_v or q75_v
            top_anchor = min(by0, wy_hi)
            v_str = f"{mean_v:.3f}"
            vb = draw.textbbox((0, 0), v_str, font=f_val)
            draw.text((wx - (vb[2] - vb[0]) // 2, top_anchor - 22 * S), v_str, font=f_val, fill=C_NAVY)

        # Group X label
        raw_lbl = en_lbl if lang == "EN" else cn_lbl
        for l_idx, line in enumerate(raw_lbl.split("\n")):
            lb = draw.textbbox((0, 0), line, font=f_tick)
            draw.text((cx - (lb[2] - lb[0]) // 2, py1 + (14 + l_idx * 24) * S), line, font=f_tick, fill=C_NAVY)

    out_img = img.resize((2400, 1120), Image.Resampling.LANCZOS)
    if lang == "EN":
        out_path = FIG_EN_DIR / "fig_EN_ood_9dim.png"
        out_img.save(str(out_path), format="PNG", optimize=True)
        out_img.save(str(AUDIT_DIR / "fig_EN_ood_9dim.png"), format="PNG", optimize=True)
        print(f"Saved Fig. S1 (EN): {out_path}")
    else:
        out_path = FIG_CN_DIR / "fig_ood_9dim.png"
        out_img.save(str(out_path), format="PNG", optimize=True)
        print(f"Saved Fig. S1 (CN): {out_path}")


def compute_s2_structured_data():
    df_ext = pd.read_csv(EXT_CORR_CSV)
    df_cnn = pd.read_csv(SMALLCNN_CSV)

    assert len(df_cnn) == 1024, f"Expected 1024 rows in smallcnn_test_rows.csv, got {len(df_cnn)}"
    assert int((~df_cnn["truncated"]).sum()) == 725
    assert int(df_cnn["truncated"].sum()) == 299

    def stats_from_df(sub_df):
        return {
            "all_1024": float(sub_df["pearson_r"].mean()),
            "full_725": float(sub_df[~sub_df["truncated"]]["pearson_r"].mean()),
            "trunc_299": float(sub_df[sub_df["truncated"]]["pearson_r"].mean()),
        }

    data = {
        "A_final_seed3": stats_from_df(df_ext[df_ext["branch"] == "A定稿seed3"]),
        "A_final_seed1": stats_from_df(df_ext[df_ext["branch"] == "A定稿seed1"]),
        "SmallCNN1D_seed1": stats_from_df(df_cnn),
        "SG51_NNLS": stats_from_df(df_ext[df_ext["branch"] == "SG(51,3)+NNLS(修正)"]),
        "Direct_NNLS": stats_from_df(df_ext[df_ext["branch"] == "直通NNLS(修正)"]),
        # Validation mean-spectrum diagnostic baseline (04_审阅核验链/13_现代基线对照_20261025/README.md):
        # only overall n=1,024 r=0.625 is defined; full/trunc are explicitly None so they are NEVER plotted by shifting!
        "MeanSpec_diag": {
            "all_1024": 0.625,
            "full_725": None,
            "trunc_299": None,
        },
    }

    # Explicit verification assertions against README and Table 3
    assert round(data["A_final_seed3"]["all_1024"], 3) == 0.945
    assert round(data["A_final_seed3"]["full_725"], 3) == 0.953
    assert round(data["A_final_seed3"]["trunc_299"], 3) == 0.928
    assert round(data["A_final_seed1"]["all_1024"], 3) == 0.945
    assert round(data["A_final_seed1"]["full_725"], 3) == 0.953
    assert round(data["A_final_seed1"]["trunc_299"], 3) == 0.927
    assert round(data["SmallCNN1D_seed1"]["all_1024"], 3) == 0.580
    assert round(data["SmallCNN1D_seed1"]["full_725"], 3) == 0.600
    assert round(data["SmallCNN1D_seed1"]["trunc_299"], 3) == 0.530
    assert round(data["SG51_NNLS"]["all_1024"], 3) == 0.916
    assert round(data["SG51_NNLS"]["full_725"], 3) == 0.920
    assert round(data["SG51_NNLS"]["trunc_299"], 3) == 0.906
    assert round(data["Direct_NNLS"]["all_1024"], 3) == 0.759
    assert round(data["Direct_NNLS"]["full_725"], 3) == 0.764
    assert round(data["Direct_NNLS"]["trunc_299"], 3) == 0.749
    return data


def render_fig_s2(lang: str = "EN"):
    data = compute_s2_structured_data()
    S = 2
    W, H = 2000 * S, 1100 * S
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if lang == "EN":
        f_title = load_font("segoeuib.ttf", 25 * S)
        f_sub = load_font("segoeuii.ttf", 17 * S)
        f_axis = load_font("segoeuib.ttf", 19 * S)
        f_tick = load_font("segoeuib.ttf", 16 * S)
        f_val = load_font("segoeuib.ttf", 15 * S)
        f_leg = load_font("segoeuib.ttf", 16 * S)
        f_note = load_font("segoeuii.ttf", 16 * S)
    else:
        f_title = load_font("msyhbd.ttc", 24 * S)
        f_sub = load_font("msyh.ttc", 16 * S)
        f_axis = load_font("msyhbd.ttc", 18 * S)
        f_tick = load_font("msyhbd.ttc", 15 * S)
        f_val = load_font("segoeuib.ttf", 15 * S)
        f_leg = load_font("msyhbd.ttc", 15 * S)
        f_note = load_font("msyh.ttc", 15 * S)

    C_NAVY = (22, 43, 76)
    C_GRID = (224, 230, 238)
    C_ALL = (44, 105, 154)
    C_FULL = (91, 141, 184)
    C_TRUNC = (232, 133, 61)
    C_DIAG = (128, 133, 142)

    if lang == "EN":
        title = "Architectural Control on the Extended-Distribution Suite (n = 1,024: 725 Complete, 299 Truncated)"
        subtitle = "Shared training stream (seed 1042), 8,000-step budget, and valid-window masked composite loss"
        ylabel = "Mean spectral Pearson correlation r"
        method_Order = [
            ("A_final_seed1", "1D-Conformer-ResUNet\n(A-final seed 1, 1.81M)\n[seed 3: 0.945 / 0.953 / 0.928]"),
            ("SmallCNN1D_seed1", "SmallCNN1D Control\n(seed 1, 0.71M)\n[single-scale residual CNN]"),
            ("MeanSpec_diag", "Validation Mean Spectrum\n(zero-info diagnostic)\n[overall n = 1,024 only]"),
            ("SG51_NNLS", "Mask-Fair SG(51,3)+NNLS\n(λ = 200)\n[classical baseline]"),
            ("Direct_NNLS", "Mask-Fair Direct NNLS\n(λ = 1000)\n[unfiltered baseline]"),
        ]
        leg_items = [
            (C_ALL, "All records (n = 1,024)"),
            (C_FULL, "Complete window (n = 725)"),
            (C_TRUNC, "Randomly truncated (n = 299)"),
        ]
        diag_line_lbl = "Validation mean-spectrum diagnostic baseline (all 1,024 records: r = 0.625)"
    else:
        title = "扩展分布测试集上的同预算卷积架构对照实验（n = 1,024：完整窗 725 条，随机截断 299 条）"
        subtitle = "相同训练数据流（种子 1042）、8,000 步预算与有效观测窗口掩码复合损失"
        ylabel = "平均谱 Pearson 相关系数 r"
        method_Order = [
            ("A_final_seed1", "1D-Conformer-ResUNet\n(A-final seed 1, 1.81M)\n[seed 3: 0.945 / 0.953 / 0.928]"),
            ("SmallCNN1D_seed1", "SmallCNN1D 对照模型\n(seed 1, 0.71M)\n[单尺度残差 CNN]"),
            ("MeanSpec_diag", "验证集平均谱诊断基线\n(零信息参照)\n[仅定义全集 n = 1,024]"),
            ("SG51_NNLS", "掩码修正 SG(51,3)+NNLS\n(λ = 200)\n[传统平滑反演基线]"),
            ("Direct_NNLS", "掩码修正直通 NNLS\n(λ = 1000)\n[无预滤波基线]"),
        ]
        leg_items = [
            (C_ALL, "全部记录 (n = 1,024)"),
            (C_FULL, "完整观测窗口 (n = 725)"),
            (C_TRUNC, "随机截断子集 (n = 299)"),
        ]
        diag_line_lbl = "验证集平均谱零信息诊断基线（全集 n = 1,024：r = 0.625）"

    tb = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) // 2, 22 * S), title, font=f_title, fill=C_NAVY)
    sb = draw.textbbox((0, 0), subtitle, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) // 2, 60 * S), subtitle, font=f_sub, fill=(80, 92, 110))

    # Legend
    leg_y = 102 * S
    lx = 360 * S
    for col, l_txt in leg_items:
        draw.rectangle((lx, leg_y, lx + 30 * S, leg_y + 20 * S), fill=col, outline=C_NAVY, width=1 * S)
        draw.text((lx + 40 * S, leg_y - 2 * S), l_txt, font=f_leg, fill=C_NAVY)
        lb = draw.textbbox((0, 0), l_txt, font=f_leg)
        lx += 40 * S + (lb[2] - lb[0]) + 60 * S

    px0, py0 = 130 * S, 152 * S
    px1, py1 = W - 56 * S, H - 155 * S
    pw, ph = px1 - px0, py1 - py0
    y_min, y_max = 0.0, 1.08

    def val_to_y(v: float) -> int:
        return py1 - int(ph * ((v - y_min) / (y_max - y_min)))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = val_to_y(tick)
        draw.line([(px0, gy), (px1, gy)], fill=C_GRID, width=2 * S)
        draw.line([(px0 - 8 * S, gy), (px0, gy)], fill=C_NAVY, width=2 * S)
        tl = f"{tick:.1f}"
        draw.text((px0 - 50 * S, gy - 11 * S), tl, font=f_tick, fill=C_NAVY)

    # Horizontal dashed reference line for validation mean-spectrum diagnostic baseline (0.625)
    y_diag = val_to_y(0.625)
    for x_d in range(px0, px1, 20 * S):
        draw.line([(x_d, y_diag), (min(x_d + 11 * S, px1), y_diag)], fill=(110, 40, 40), width=2 * S)

    draw.line([(px0, py0), (px0, py1), (px1, py1)], fill=C_NAVY, width=3 * S)
    draw.text((24 * S, py0 - 34 * S), ylabel, font=f_axis, fill=C_NAVY)

    n_m = len(method_Order)
    g_step = pw / n_m
    bw = 56 * S
    sub_specs = [
        ("all_1024", -bw, C_ALL),
        ("full_725", 0, C_FULL),
        ("trunc_299", bw, C_TRUNC),
    ]

    for m_idx, (m_key, m_label) in enumerate(method_Order):
        cx = px0 + int((m_idx + 0.5) * g_step)
        m_dict = data[m_key]
        if m_key == "MeanSpec_diag":
            # Only plot the single overall bar centered at cx
            v = m_dict["all_1024"]
            bx0, bx1 = int(cx - bw * 0.55), int(cx + bw * 0.55)
            by0 = val_to_y(v)
            draw.rectangle((bx0, by0, bx1, py1), fill=C_DIAG, outline=C_NAVY, width=2 * S)
            v_str = f"{v:.3f}"
            vb = draw.textbbox((0, 0), v_str, font=f_val)
            draw.text((cx - (vb[2] - vb[0]) // 2, by0 - 28 * S), v_str, font=f_val, fill=C_NAVY)
            # Place diagnostic baseline banner cleanly in the open space above the 0.625 bar
            db = draw.textbbox((0, 0), diag_line_lbl, font=f_note)
            dw = db[2] - db[0]
            dh = db[3] - db[1]
            dx0 = cx - dw // 2 - 12 * S
            dy0 = by0 - 68 * S
            draw.rounded_rectangle((dx0, dy0, dx0 + dw + 24 * S, dy0 + dh + 12 * S),
                                   radius=6 * S, fill=(255, 248, 248), outline=(160, 70, 70), width=1 * S)
            draw.text((cx - dw // 2, dy0 + 4 * S), diag_line_lbl, font=f_note, fill=(110, 40, 40))
        else:
            for sub_key, x_off, col in sub_specs:
                v = m_dict[sub_key]
                assert v is not None, f"Unexpected None for {m_key}.{sub_key}"
                bx0 = int(cx + x_off - bw * 0.44)
                bx1 = int(cx + x_off + bw * 0.44)
                by0 = val_to_y(v)
                draw.rectangle((bx0, by0, bx1, py1), fill=col, outline=C_NAVY, width=2 * S)
                v_str = f"{v:.3f}"
                vb = draw.textbbox((0, 0), v_str, font=f_val)
                draw.text(((bx0 + bx1 - (vb[2] - vb[0])) // 2, by0 - 26 * S), v_str, font=f_val, fill=C_NAVY)

        for l_idx, line in enumerate(m_label.split("\n")):
            fn = f_tick if l_idx < 2 else f_note
            col_t = C_NAVY if l_idx < 2 else (90, 100, 115)
            lb = draw.textbbox((0, 0), line, font=fn)
            draw.text((cx - (lb[2] - lb[0]) // 2, py1 + (14 + l_idx * 26) * S), line, font=fn, fill=col_t)

    out_img = img.resize((2000, 1100), Image.Resampling.LANCZOS)
    if lang == "EN":
        out_path = FIG_EN_DIR / "fig_EN_cnn_compare.png"
        out_img.save(str(out_path), format="PNG", optimize=True)
        out_img.save(str(AUDIT_DIR / "fig_EN_cnn_compare.png"), format="PNG", optimize=True)
        print(f"Saved Fig. S2 (EN): {out_path}")
    else:
        out_path = FIG_CN_DIR / "fig_cnn_compare.png"
        out_img.save(str(out_path), format="PNG", optimize=True)
        print(f"Saved Fig. S2 (CN): {out_path}")


def patch_generate_en_figures_script():
    script_path = ROOT / "01_论文正文" / "图表脚本" / "generate_en_figures.py"
    dispatcher_code = '''# -*- coding: utf-8 -*-
r"""
Unified, verified entry-point for regenerating JMR submission figures.
Delegates directly to the audited key-bound figure generators in
04_审阅核验链/22_Gemini独立审计_20261106/ so that legacy hardcoded arrays or
external workspace paths can never overwrite the validated manuscript figures.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106"
sys.path.insert(0, str(AUDIT_DIR))

import regenerate_fig3_clean
import regenerate_fig6_clean
import regenerate_graphical_abstract
import regenerate_supp_figures_clean


def main():
    regenerate_fig3_clean.main()
    regenerate_fig6_clean.render_fig6("EN")
    regenerate_fig6_clean.render_fig6("CN")
    regenerate_supp_figures_clean.render_fig_s1("EN")
    regenerate_supp_figures_clean.render_fig_s1("CN")
    regenerate_supp_figures_clean.render_fig_s2("EN")
    regenerate_supp_figures_clean.render_fig_s2("CN")
    regenerate_graphical_abstract.render_ga()
    print("All verified JMR figures regenerated from audited sources.")


if __name__ == "__main__":
    main()
'''
    script_path.write_text(dispatcher_code, encoding="utf-8")
    print("Replaced legacy generate_en_figures.py with verified audited figure dispatcher")


if __name__ == "__main__":
    patch_generate_en_figures_script()
    render_fig_s1("EN")
    render_fig_s1("CN")
    render_fig_s2("EN")
    render_fig_s2("CN")
