# -*- coding: utf-8 -*-
r"""
Regenerate Fig. 6 (Component-wise timing comparison across separate benchmarks):
- English: 01_论文正文/figures_EN/fig_EN_timing_budget.png
- Chinese: 01_论文正文/figures/fig_timing_budget.png

Strictly separates:
1. Nominal active CPMG acquisition window (N * TE: 4.00 s, 3.00 s, 1.80 s)
2. Separate fixed-input on-board compute benchmark (08_板端证据/board_benchmark_real_20261024/log_m5_single.txt & log_m5_omp.txt):
   - Single-core ARM (66.8 ms/frame): 26.11 s (4,000 echoes), 19.57 s (3,000 echoes), 11.78 s (1,800 echoes)
   - Dual-core ARM OpenMP (35.9 ms/frame): 14.04 s (4,000 echoes), 10.52 s (3,000 echoes), 6.32 s (1,800 echoes)
3. 15-record file-replay test harness (08_板端证据/板端回放_20261024/data_down/replay_timing.log, ~43.45 ms/frame):
   - ~4,000 echoes (200,014 pts, 391 frames): 16.984–16.998 s (mean 16.99 s)
   - ~3,000 echoes (150,014 pts, 293 frames): 12.738–12.765 s (mean 12.75 s)
   - ~1,800 echoes (90,006 pts, 176 frames):  7.644–7.665 s (mean 7.65 s)
4. On-board matched dual-channel SG(21,2) compute benchmark (0.179 s = 178.8 ms for 4,000 echoes, 1 thread)
5. Host PC 2-thread fixed-input inversion benchmark (04_审阅核验链/04_审阅与证据链/review_20261024/additional_checks.json):
   - Historical A-original (1D-Conformer-ResUNet) median: 14.55 ms
   - Classical SG+NNLS median: 7.91 ms
6. Removes the unverified 'GbE transfer (16 KB), 1.2 ms' bar.
"""
import json
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LOG_OMP = ROOT / "08_板端证据" / "board_benchmark_real_20261024" / "log_m5_omp.txt"
LOG_SINGLE = ROOT / "08_板端证据" / "board_benchmark_real_20261024" / "log_m5_single.txt"
LOG_REPLAY = ROOT / "08_板端证据" / "板端回放_20261024" / "data_down" / "replay_timing.log"
HOST_JSON = ROOT / "04_审阅核验链" / "04_审阅与证据链" / "review_20261024" / "additional_checks.json"

FIG_EN = ROOT / "01_论文正文" / "figures_EN" / "fig_EN_timing_budget.png"
FIG_CN = ROOT / "01_论文正文" / "figures" / "fig_timing_budget.png"
AUDIT_DIR = ROOT / "04_审阅核验链" / "22_Gemini独立审计_20261106"
FONT_DIR = Path(r"C:\Windows\Fonts")


def load_font(name: str, size: int):
    p = FONT_DIR / name
    if p.exists():
        return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def verify_and_load_timing_data():
    omp_txt = LOG_OMP.read_text(encoding="utf-8")
    single_txt = LOG_SINGLE.read_text(encoding="utf-8")
    replay_txt = LOG_REPLAY.read_text(encoding="utf-8")
    host_data = json.loads(HOST_JSON.read_text(encoding="utf-8"))

    # Verify fixed-input compute benchmark numbers from archived logs
    assert "6.321 秒" in omp_txt and "10.521 秒" in omp_txt and "14.036 秒" in omp_txt
    assert "11.775 秒" in single_txt and "19.574 秒" in single_txt and "26.112 秒" in single_txt

    # Verify 15-record replay timing log ranges
    replay_4000 = [float(m.group(1)) for m in re.finditer(r"n_pts=200014 .*? secs=([0-9.]+)", replay_txt)]
    replay_3000 = [float(m.group(1)) for m in re.finditer(r"n_pts=150014 .*? secs=([0-9.]+)", replay_txt)]
    replay_1800 = [float(m.group(1)) for m in re.finditer(r"n_pts=90006 .*? secs=([0-9.]+)", replay_txt)]
    assert len(replay_4000) == 5 and len(replay_3000) == 6 and len(replay_1800) == 4
    assert round(min(replay_4000), 3) == 16.984 and round(max(replay_4000), 3) == 16.998
    assert round(min(replay_3000), 3) == 12.738 and round(max(replay_3000), 3) == 12.765
    assert round(min(replay_1800), 3) == 7.644 and round(max(replay_1800), 3) == 7.665

    # Verify host PC fixed-input timing medians
    a_host_ms = host_data["cpu_timing_single_fixed_input_2threads"]["A_Conformer"]["median_ms"]
    sg_host_ms = host_data["cpu_timing_single_fixed_input_2threads"]["SG_NNLS"]["median_ms"]
    assert round(a_host_ms, 2) == 14.55
    assert round(sg_host_ms, 2) == 7.91

    return {
        "echo_groups": [4000, 3000, 1800],
        "nominal_window_s": {4000: 4.00, 3000: 3.00, 1800: 1.80},
        "arm_single_compute_s": {4000: 26.112, 3000: 19.574, 1800: 11.775},
        "arm_dual_compute_s": {4000: 14.036, 3000: 10.521, 1800: 6.321},
        "arm_dual_replay_mean_s": {
            4000: sum(replay_4000) / len(replay_4000),
            3000: sum(replay_3000) / len(replay_3000),
            1800: sum(replay_1800) / len(replay_1800),
        },
        "sg21_board_ms": 178.75,  # 0.179 s (two-run median of 178.9 and 178.6 ms)
        "host_a_orig_ms": a_host_ms,
        "host_sg_nnls_ms": sg_host_ms,
    }


def render_fig6(lang: str = "EN"):
    data = verify_and_load_timing_data()
    S = 2
    W, H = 2400 * S, 1080 * S
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if lang == "EN":
        f_title = load_font("segoeuib.ttf", 24 * S)
        f_sub = load_font("segoeuii.ttf", 16 * S)
        f_ptitle = load_font("segoeuib.ttf", 20 * S)
        f_axis = load_font("segoeuib.ttf", 17 * S)
        f_tick = load_font("segoeuib.ttf", 15 * S)
        f_val = load_font("segoeuib.ttf", 14 * S)
        f_leg = load_font("segoeuib.ttf", 14 * S)
        f_note = load_font("segoeui.ttf", 14 * S)
    else:
        f_title = load_font("msyhbd.ttc", 23 * S)
        f_sub = load_font("msyh.ttc", 15 * S)
        f_ptitle = load_font("msyhbd.ttc", 19 * S)
        f_axis = load_font("msyhbd.ttc", 16 * S)
        f_tick = load_font("msyhbd.ttc", 14 * S)
        f_val = load_font("segoeuib.ttf", 14 * S)
        f_leg = load_font("msyhbd.ttc", 14 * S)
        f_note = load_font("msyh.ttc", 14 * S)

    C_NAVY = (22, 43, 76)
    C_GRID = (224, 230, 238)
    C_ACQ = (127, 140, 141)       # Gray for nominal active acquisition window
    C_SINGLE = (192, 57, 43)     # Red for single-core compute benchmark
    C_DUAL = (44, 105, 154)      # Blue for dual-core compute benchmark
    C_REPLAY = (230, 126, 34)    # Orange for 15-record file-replay harness
    C_SG = (39, 174, 96)         # Green for on-board SG(21,2)
    C_HOST = (142, 68, 173)      # Purple for host PC benchmarks

    if lang == "EN":
        sup_title = "Component-Wise Timing Comparison Across Separate Benchmarks (Not Online End-to-End Latency)"
        sup_sub = "Separating 15-record file-replay harness times (~43.45 ms/frame), fixed-input ARM compute benchmarks (35.9 / 66.8 ms/frame), and host 2-thread benchmarks"
    else:
        sup_title = "独立测试下的分项时间对比（区分文件回放测试程序与固定输入计算基准，非在线端到端实测）"
        sup_sub = "严格区分 15 条实测记录回放测试程序（~43.45 ms/帧）、板端固定输入计算基准（35.9 / 66.8 ms/帧）与上位机双线程基准"

    tb = draw.textbbox((0, 0), sup_title, font=f_title)
    draw.text(((W - (tb[2] - tb[0])) // 2, 18 * S), sup_title, font=f_title, fill=C_NAVY)
    sb = draw.textbbox((0, 0), sup_sub, font=f_sub)
    draw.text(((W - (sb[2] - sb[0])) // 2, 52 * S), sup_sub, font=f_sub, fill=(90, 100, 115))

    # =========================================================================
    # Panel (a): Left half — Scaling across 4,000 / 3,000 / 1,800 echoes
    # =========================================================================
    pa_x0, pa_y0, pa_x1, pa_y1 = 110 * S, 185 * S, 1100 * S, 860 * S
    if lang == "EN":
        pa_title = "(a) Echo-Count Scaling: Nominal Window vs. ARM Benchmarks"
        pa_ylabel = "Time (s)"
        group_labels = [
            "4,000 echoes\n(200k pts, 391 frames)",
            "3,000 echoes\n(150k pts, 293 frames)",
            "1,800 echoes\n(90k pts, 176 frames)",
        ]
        leg_a = [
            (C_ACQ, "Nominal active acquisition window (N × TE)"),
            (C_SINGLE, "ARM single-core compute benchmark (66.8 ms/frame)"),
            (C_DUAL, "ARM dual-core OpenMP compute benchmark (35.9 ms/frame)"),
            (C_REPLAY, "15-record file-replay harness (~43.45 ms/frame, 2 threads)"),
        ]
        note_a = "Note: Compute benchmarks use fixed-seed noisy inputs (log_m5_*.txt); replay harness includes file replay overhead (replay_timing.log)."
    else:
        pa_title = "(a) 不同回波数下的名义采集窗与 ARM 板端耗时对比"
        pa_ylabel = "时间 (s)"
        group_labels = [
            "4,000 回波\n(200k 点, 391 帧)",
            "3,000 回波\n(150k 点, 293 帧)",
            "1,800 回波\n(90k 点, 176 帧)",
        ]
        leg_a = [
            (C_ACQ, "名义有效采集窗 (N × TE)"),
            (C_SINGLE, "ARM 单核固定输入计算基准 (66.8 ms/帧)"),
            (C_DUAL, "ARM 双核 OpenMP 固定输入计算基准 (35.9 ms/帧)"),
            (C_REPLAY, "15 条实测记录文件回放测试程序 (~43.45 ms/帧, 双线程)"),
        ]
        note_a = "注：计算基准采用固定种子带噪输入 (log_m5_*.txt)；回放测试程序含文件回放流程开销 (replay_timing.log)。"

    ptb = draw.textbbox((0, 0), pa_ptitle := pa_title, font=f_ptitle)
    draw.text(((pa_x0 + pa_x1 - (ptb[2] - ptb[0])) // 2, 100 * S), pa_ptitle, font=f_ptitle, fill=C_NAVY)
    draw.text((pa_x0 - 50 * S, pa_y0 - 36 * S), pa_ylabel, font=f_axis, fill=C_NAVY)

    y_max_a = 35.0
    def val_to_ya(v: float) -> int:
        return int(pa_y1 - (v / y_max_a) * (pa_y1 - pa_y0))

    for y_tick in [0, 5, 10, 15, 20, 25, 30, 35]:
        yy = val_to_ya(y_tick)
        draw.line((pa_x0, yy, pa_x1, yy), fill=C_GRID, width=2 * S)
        lbl = f"{y_tick}"
        lb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((pa_x0 - (lb[2] - lb[0]) - 12 * S, yy - (lb[3] - lb[1]) // 2), lbl, font=f_tick, fill=C_NAVY)

    draw.line((pa_x0, pa_y0, pa_x0, pa_y1), fill=C_NAVY, width=3 * S)
    draw.line((pa_x0, pa_y1, pa_x1, pa_y1), fill=C_NAVY, width=3 * S)

    g_span = (pa_x1 - pa_x0) / 3.0
    bw = g_span * 0.18
    offsets = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]

    for g_idx, (echo_cnt, g_lbl) in enumerate(zip(data["echo_groups"], group_labels)):
        cx = pa_x0 + (g_idx + 0.5) * g_span
        vals_a = [
            (data["nominal_window_s"][echo_cnt], C_ACQ, f"{data['nominal_window_s'][echo_cnt]:.2f}s"),
            (data["arm_single_compute_s"][echo_cnt], C_SINGLE, f"{data['arm_single_compute_s'][echo_cnt]:.2f}s"),
            (data["arm_dual_compute_s"][echo_cnt], C_DUAL, f"{data['arm_dual_compute_s'][echo_cnt]:.2f}s"),
            (data["arm_dual_replay_mean_s"][echo_cnt], C_REPLAY, f"{data['arm_dual_replay_mean_s'][echo_cnt]:.2f}s"),
        ]
        for off, (v, col, v_str) in zip(offsets, vals_a):
            bx0 = int(cx + off - bw * 0.44)
            bx1 = int(cx + off + bw * 0.44)
            by0 = val_to_ya(v)
            draw.rectangle((bx0, by0, bx1, pa_y1), fill=col, outline=C_NAVY, width=2 * S)
            vb = draw.textbbox((0, 0), v_str, font=f_val)
            draw.text(((bx0 + bx1 - (vb[2] - vb[0])) // 2, by0 - 24 * S), v_str, font=f_val, fill=C_NAVY)

        for l_idx, line in enumerate(g_lbl.split("\n")):
            lb = draw.textbbox((0, 0), line, font=f_tick)
            draw.text((cx - (lb[2] - lb[0]) // 2, pa_y1 + (14 + l_idx * 24) * S), line, font=f_tick, fill=C_NAVY)

    # Legend inside top-right of Panel (a) (above 3,000 and 1,800 echo bars, completely clear of 26.11s)
    lx0, ly0 = pa_x0 + 380 * S, pa_y0 + 10 * S
    draw.rectangle((lx0 - 10 * S, ly0 - 6 * S, pa_x1 - 10 * S, ly0 + 114 * S), fill=(248, 250, 253), outline=C_GRID, width=2 * S)
    for idx_l, (col, l_txt) in enumerate(leg_a):
        yy = ly0 + idx_l * 27 * S
        draw.rectangle((lx0, yy + 3 * S, lx0 + 22 * S, yy + 20 * S), fill=col, outline=C_NAVY, width=1 * S)
        draw.text((lx0 + 30 * S, yy), l_txt, font=f_leg, fill=C_NAVY)

    draw.text((pa_x0, pa_y1 + 78 * S), note_a, font=f_note, fill=(80, 90, 105))

    # =========================================================================
    # Panel (b): Right half — Component-wise timing budget (4,000-echo reference)
    # =========================================================================
    pb_x0, pb_y0, pb_x1, pb_y1 = 1580 * S, 185 * S, 2320 * S, 860 * S
    if lang == "EN":
        pb_title = "(b) Component-Wise Timing Budget (4,000-Echo Reference)"
        pb_xlabel = "Measured Time on Log10 Scale (ms)"
        bars_b = [
            ("Nominal active window\n(N × TE, 4,000 × 1.0 ms)", 4000.0, C_ACQ, "4,000 ms (4.00 s)"),
            ("On-board Edge ECA-DS-TCN\n(15-record replay harness, 2T)", 16992.0, C_REPLAY, "16,992 ms (16.99 s)"),
            ("On-board Edge ECA-DS-TCN\n(fixed-input compute benchmark, 2T)", 14036.0, C_DUAL, "14,036 ms (14.04 s)"),
            ("On-board matched SG(21,2)\n(dual-channel compute benchmark, 1T)", 178.8, C_SG, "178.8 ms (0.179 s)"),
            ("Host PC 1D-Conformer-ResUNet\n(A-original fixed-input median, 2T)", 14.55, C_HOST, "14.55 ms"),
            ("Host PC SG+NNLS inversion\n(fixed-input median, 2T)", 7.91, C_HOST, "7.91 ms"),
        ]
        note_b = "Note: Component times come from separate benchmarks and do not represent concurrent online pipeline execution."
    else:
        pb_title = "(b) 4,000 回波分项时间对照（来自独立测试）"
        pb_xlabel = "对数坐标时间 Log10 (ms)"
        bars_b = [
            ("名义有效采集窗\n(N × TE, 4,000 × 1.0 ms)", 4000.0, C_ACQ, "4,000 ms (4.00 s)"),
            ("板端 Edge ECA-DS-TCN\n(15 条记录回放测试程序, 双线程)", 16992.0, C_REPLAY, "16,992 ms (16.99 s)"),
            ("板端 Edge ECA-DS-TCN\n(固定输入计算基准, 双线程)", 14036.0, C_DUAL, "14,036 ms (14.04 s)"),
            ("板端同配置 SG(21,2)\n(双通道计算基准, 单线程)", 178.8, C_SG, "178.8 ms (0.179 s)"),
            ("上位机 1D-Conformer-ResUNet\n(历史 A-original 固定输入中位数, 双线程)", 14.55, C_HOST, "14.55 ms"),
            ("上位机 SG+NNLS 反演\n(固定输入中位数, 双线程)", 7.91, C_HOST, "7.91 ms"),
        ]
        note_b = "注：各分项来自独立基准测试，不含未单独测量的网络传输项，不代表在线流水线端到端实测延迟。"

    ptb_b = draw.textbbox((0, 0), pb_title, font=f_ptitle)
    draw.text(((1220 * S + pb_x1 - (ptb_b[2] - ptb_b[0])) // 2, 100 * S), pb_title, font=f_ptitle, fill=C_NAVY)

    log_min, log_max = 0.0, 5.0  # 10^0 = 1 ms to 10^5 = 100,000 ms
    def ms_to_xb(ms_val: float) -> int:
        lv = math.log10(max(ms_val, 1.0))
        return int(pb_x0 + ((lv - log_min) / (log_max - log_min)) * (pb_x1 - pb_x0))

    for exp in range(0, 6):
        xx = ms_to_xb(10 ** exp)
        draw.line((xx, pb_y0, xx, pb_y1), fill=C_GRID, width=2 * S)
        lbl = f"10^{exp}" if exp >= 2 else ("1" if exp == 0 else "10")
        lb = draw.textbbox((0, 0), lbl, font=f_tick)
        draw.text((xx - (lb[2] - lb[0]) // 2, pb_y1 + 12 * S), lbl, font=f_tick, fill=C_NAVY)

    draw.line((pb_x0, pb_y0, pb_x0, pb_y1), fill=C_NAVY, width=3 * S)
    draw.line((pb_x0, pb_y1, pb_x1, pb_y1), fill=C_NAVY, width=3 * S)

    row_span = (pb_y1 - pb_y0) / len(bars_b)
    bh = row_span * 0.54
    for r_idx, (b_lbl, ms_val, col, val_txt) in enumerate(bars_b):
        cy = pb_y0 + (r_idx + 0.5) * row_span
        by0 = int(cy - bh / 2)
        by1 = int(cy + bh / 2)
        bx1 = ms_to_xb(ms_val)
        draw.rectangle((pb_x0, by0, bx1, by1), fill=col, outline=C_NAVY, width=2 * S)

        # Value label to the right of bar
        vb = draw.textbbox((0, 0), val_txt, font=f_val)
        draw.text((bx1 + 10 * S, cy - (vb[3] - vb[1]) // 2), val_txt, font=f_val, fill=C_NAVY)

        # Row label to the left of pb_x0
        lines = b_lbl.split("\n")
        for l_i, line in enumerate(lines):
            lb = draw.textbbox((0, 0), line, font=f_tick)
            y_off = cy - (len(lines) * 20 * S) // 2 + l_i * 22 * S
            draw.text((pb_x0 - (lb[2] - lb[0]) - 14 * S, y_off), line, font=f_tick, fill=C_NAVY)

    xb_lbl = draw.textbbox((0, 0), pb_xlabel, font=f_axis)
    draw.text(((pb_x0 + pb_x1 - (xb_lbl[2] - xb_lbl[0])) // 2, pb_y1 + 44 * S), pb_xlabel, font=f_axis, fill=C_NAVY)
    draw.text((1200 * S, pb_y1 + 78 * S), note_b, font=f_note, fill=(80, 90, 105))

    out_img = img.resize((2400, 1080), Image.Resampling.LANCZOS)
    if lang == "EN":
        out_img.save(str(FIG_EN), format="PNG", optimize=True)
        out_img.save(str(AUDIT_DIR / "fig_EN_timing_budget.png"), format="PNG", optimize=True)
        print(f"Saved Fig. 6 (EN): {FIG_EN}")
    else:
        out_img.save(str(FIG_CN), format="PNG", optimize=True)
        print(f"Saved Fig. 6 (CN): {FIG_CN}")


if __name__ == "__main__":
    render_fig6("EN")
    render_fig6("CN")
