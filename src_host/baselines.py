# -*- coding: utf-8 -*-
"""
baselines.py —— 传统方法基线与统一指标

基线(与 Conformer 完全同输入同协议):
  * NNLS-Tikhonov 反演(二阶差分正则, 与 m7 传统反演同型, 前向核统一为 KERNEL_AW);
  * SG 平滑 + NNLS(SG 仅作用于包络, 参数在开发集上选定后冻结)。

指标(全部作用于 128 点对数谱, 输入/参考统一为单位 log 面积):
  * pearson_r            谱形皮尔逊相关
  * peak_err_log         主峰位误差 |log10 T2_pred - log10 T2_ref|
                         (argmax + 三点抛物线内插, 避免落入同一格即记 0 的假精度)
  * spec_l2              归一化谱 L2 距离
  * fwd_rmse             前向自洽 ||A_w@m_scaled - y_norm||, 幅度对齐后计算
  * area_frac_err        短/长 T2 组分面积分数误差(阈值 60 ms / 300 ms 两侧)
"""

import numpy as np
from scipy.optimize import nnls
from scipy.signal import savgol_filter

from hconfig import STD_LOG_T2, STD_T2_MS, KERNEL_AW, LOG_GRID_W


# ---------------------------------------------------------------- 基线方法
def nnls_invert(echo_norm, reg_lambda=2.0, valid_mask=None):
    """Tikhonov-NNLS: 输入单位化包络(首点≈1), 输出单位 log 面积谱。

    2026-10-25 修正(10_独立审阅 P0): valid_mask 提供时, 数据一致性项只使用
    有效观测行 —— 补零的缺失观测不再被当作"测到的零"参与拟合。默认 None
    时行为与历史版本逐位一致(全观测场景不受影响)。"""
    if valid_mask is not None:
        rows = np.where(np.asarray(valid_mask) > 0.5)[0]
        K = KERNEL_AW[rows, :]
        y = np.asarray(echo_norm, dtype=np.float64)[rows]
    else:
        K = KERNEL_AW
        y = np.asarray(echo_norm, dtype=np.float64)
    n_bins = K.shape[1]
    D2 = np.zeros((n_bins - 2, n_bins), dtype=np.float64)
    idx = np.arange(n_bins - 2)
    D2[idx, idx] = 1.0
    D2[idx, idx + 1] = -2.0
    D2[idx, idx + 2] = 1.0
    reg_w = np.sqrt(reg_lambda) * (np.linalg.norm(K, "fro") / np.linalg.norm(D2, "fro"))
    K_aug = np.vstack([K, reg_w * D2])

    scale = max(float(np.max(np.abs(y))), 1e-6)
    y_aug = np.concatenate([y / scale, np.zeros(n_bins - 2)])
    spec, _ = nnls(K_aug, y_aug)
    area = float(np.sum(spec) * LOG_GRID_W)
    if area > 1e-9:
        spec = spec / area
    return spec


def sg_denoise(echo_norm, win=21, poly=2, valid_mask=None):
    """SG 平滑; valid_mask 提供时只在有效前缀内滤波, 不跨补零边界。"""
    x = np.asarray(echo_norm, dtype=np.float64)
    if valid_mask is None:
        return savgol_filter(x, win, poly)
    n_valid = int(np.sum(np.asarray(valid_mask) > 0.5))
    out = x.copy()
    if n_valid >= win:
        out[:n_valid] = savgol_filter(x[:n_valid], win, poly)
    return out


def sg_nnls_invert(echo_norm, win=21, poly=2, reg_lambda=2.0, valid_mask=None):
    return nnls_invert(sg_denoise(echo_norm, win, poly, valid_mask),
                       reg_lambda, valid_mask)


# ---------------------------------------------------------------- 统一指标
def peak_pos_log(spec):
    """argmax + 对数域三点抛物线内插的主峰位置(log10 T2)。"""
    i = int(np.argmax(spec))
    if 0 < i < len(spec) - 1:
        y0, y1, y2 = float(spec[i - 1]), float(spec[i]), float(spec[i + 1])
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
    else:
        delta = 0.0
    return STD_LOG_T2[i] + delta * LOG_GRID_W


def pearson_r(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def area_fraction(spec, t2_threshold_ms):
    """谱在 T2 < t2_threshold_ms 一侧的面积分数(连续谱, 对数面积归一)。

    修复记录(2026-10-24): 此前误写 np.exp(STD_LOG_T2) —— STD_LOG_T2 是
    log10 网格, exp 后最大仅约 54.6, 使 60/300ms 两个阈值掩码都选中全部
    128 点, 面积分数恒为 1, 报告中两列面积误差恒为 0(虚假完美)。
    正确应为 10**STD_LOG_T2, 即直接用 STD_T2_MS。"""
    mask = STD_T2_MS < t2_threshold_ms
    return float(np.sum(spec[mask]) * LOG_GRID_W)


def spectrum_metrics(pred, ref, echo_norm=None, valid_mask=None):
    """计算统一指标集。echo_norm 提供时才计算 fwd_rmse;
    valid_mask(0/1数组)提供时前向RMSE只在有效时间窗内计算(截断样本)。"""
    p_area = float(np.sum(pred) * LOG_GRID_W)
    p = pred / p_area if p_area > 1e-9 else pred
    r_area = float(np.sum(ref) * LOG_GRID_W)
    r = ref / r_area if r_area > 1e-9 else ref

    out = {
        "pearson_r": pearson_r(p, r),
        "peak_err_log": abs(peak_pos_log(p) - peak_pos_log(r)),
        "spec_l2": float(np.linalg.norm(p - r)),
        "area_short_err": abs(area_fraction(p, 60.0) - area_fraction(r, 60.0)),
        "area_long_err": abs(area_fraction(p, 300.0) - area_fraction(r, 300.0)),
    }
    if echo_norm is not None:
        idx = np.asarray(valid_mask) > 0.5 if valid_mask is not None else slice(None)
        y_fwd = KERNEL_AW @ pred
        e = np.asarray(echo_norm)[idx]
        c = float(e[0]) / max(float((KERNEL_AW @ pred)[idx][0]), 1e-9)
        out["fwd_rmse"] = float(np.sqrt(np.mean((y_fwd[idx] * c - e) ** 2)))
    return out


# ------------------------------------------------- 开发集参数选择(先冻结)
def select_classical_params(dev_echoes, dev_specs, log_fn=print):
    """在开发集(固定仿真验证集, 与测试集种子不相交)上选择传统参数, 选定后全程冻结。

    整改记录(2026-10-24 依据独立评估意见):
      * 旧网格 λ∈{0.5..10} 选中上界 10, 不充分 —— 扩展到 λ∈{0.5,2,10,50,200,1000};
      * 旧实现 SG 参数在先选定的 λ 上单独挑, 不是联合选择 —— 改为对每个
        (λ, win, poly) 组合联合打分;
      * 直通NNLS 与 SG+NNLS 分支分别冻结各自最优 λ(两者最优可不同)。
    返回 (sg_win, sg_poly, nnls_lambda_sg, nnls_lambda_direct)。"""
    LAMBDAS = (0.5, 2.0, 10.0, 50.0, 200.0, 1000.0)

    best_direct = {"score": -1e18, "lambda": 2.0}
    for lam in LAMBDAS:
        rs = [pearson_r(nnls_invert(e, lam), s) for e, s in zip(dev_echoes, dev_specs)]
        mean_r = float(np.mean(rs))
        if mean_r > best_direct["score"]:
            best_direct = {"score": mean_r, "lambda": lam}
        log_fn(f"  [参数选择] 直通NNLS lambda={lam:<6} 平均谱相关 r={mean_r:.4f}")

    best_joint = {"score": -1e18, "win": 21, "poly": 2, "lambda": 2.0}
    for lam in LAMBDAS:
        for win in (11, 21, 31, 51, 71):
            for poly in (2, 3):
                rs = [pearson_r(sg_nnls_invert(e, win, poly, lam), s)
                      for e, s in zip(dev_echoes, dev_specs)]
                mean_r = float(np.mean(rs))
                if mean_r > best_joint["score"]:
                    best_joint = {"score": mean_r, "win": win, "poly": poly, "lambda": lam}
        log_fn(f"  [参数选择] SG×lambda={lam:<6} 网格完成, 当前联合最优 "
               f"win={best_joint['win']},poly={best_joint['poly']},"
               f"λ={best_joint['lambda']} r={best_joint['score']:.4f}")

    log_fn(f"  [参数选择] 冻结: 直通NNLS λ={best_direct['lambda']} "
           f"(r={best_direct['score']:.4f}); SG+NNLS 联合最优 "
           f"win={best_joint['win']}, poly={best_joint['poly']}, "
           f"λ={best_joint['lambda']} (r={best_joint['score']:.4f})")
    return (best_joint["win"], best_joint["poly"],
            best_joint["lambda"], best_direct["lambda"])


def select_classical_params_masked(dev_echoes, dev_specs, dev_masks,
                                   log_fn=print):
    """带有效观测窗的联合选参(2026-10-25 P1 修正): 与无 mask 版网格相同,
    但每条样本传入其 valid_mask。返回 (win, poly, lambda_sg, lambda_direct)。"""
    LAMBDAS = (0.5, 2.0, 10.0, 50.0, 200.0, 1000.0)
    best_direct = {"score": -1e18, "lambda": 2.0}
    for lam in LAMBDAS:
        rs = [pearson_r(nnls_invert(e, lam, m), s)
              for e, s, m in zip(dev_echoes, dev_specs, dev_masks)]
        r = float(np.mean(rs))
        if r > best_direct["score"]:
            best_direct = {"score": r, "lambda": lam}
        log_fn(f"  [参数选择-mask] 直通NNLS lambda={lam:<6} r={r:.4f}")
    best_joint = {"score": -1e18, "win": 21, "poly": 2, "lambda": 2.0}
    for lam in LAMBDAS:
        for win in (11, 21, 31, 51, 71):
            for poly in (2, 3):
                rs = [pearson_r(
                    sg_nnls_invert(e, win, poly, lam, m), s)
                    for e, s, m in zip(dev_echoes, dev_specs, dev_masks)]
                r = float(np.mean(rs))
                if r > best_joint["score"]:
                    best_joint = {"score": r, "win": win, "poly": poly,
                                  "lambda": lam}
        log_fn(f"  [参数选择-mask] SG×lambda={lam:<6} 完, 最优 "
               f"win={best_joint['win']},poly={best_joint['poly']},"
               f"λ={best_joint['lambda']} r={best_joint['score']:.4f}")
    log_fn(f"  [参数选择-mask] 冻结: 直通 λ={best_direct['lambda']}; "
           f"SG({best_joint['win']},{best_joint['poly']})+λ={best_joint['lambda']}")
    return (best_joint["win"], best_joint["poly"],
            best_joint["lambda"], best_direct["lambda"])
