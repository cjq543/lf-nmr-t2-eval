# -*- coding: utf-8 -*-
"""
脚本名称: m4_generate_cpmg_multiecho_dataset.py
功能:
  【阶段三整改落地方案 1：重构仿真数据生成目标 (归一化无量纲物理对齐版)】
  1. 彻底去除第 63 行硬编码的 80% 平底底座 (原 clean_envelope = macro_decay * (1.0 + 0.25 * echo_modulation))；
  2. 真实核磁 CPMG 物理回波建模：
     - 回波间距 1.0 ms (50 kSPS 下每周期 50 点)；
     - 峰顶位于周期内的 0.76 ms 处 (点 38)；
     - 物理凹槽对称位于两峰中心 0.27 ms 处 (点 13~14，半高宽约 40 us)；
     - 谷底回归自然低电平 (< 5% 峰高)，严禁人为垫高虚假底座；
  3. 成对生成目标噪声残差 (Noise Target) 与候选有效性掩码 (valid_mask)；
  4. 采用无量纲归一化振幅体系 (峰值基准归一为 1.0，SNR 跨度覆盖 8dB ~ 30dB，微弱直流偏移 0.005)。
"""

import os
import sys
import numpy as np

def generate_cpmg_multiecho_dataset(
    n_train: int = 3500,
    n_val: int = 600,
    n_test: int = 400,
    n_points: int = 1024,
    sample_rate: float = 50000.0,
    save_path: str = None,
    seed: int = 42
):
    np.random.seed(seed)
    total_samples = n_train + n_val + n_test
    print(f"[阶段 3 重构] 开始生成归一化无量纲 CPMG 残差去噪数据集 (总样本数: {total_samples})...")

    dt = 1.0 / sample_rate
    t = np.arange(n_points) * dt  # 0 ~ 20.48 ms

    X_all = np.zeros((total_samples, 2, n_points), dtype=np.float32)       # 含噪信号 (归一化 ~1.0)
    Y_clean_all = np.zeros((total_samples, 2, n_points), dtype=np.float32) # 纯净信号
    Y_noise_all = np.zeros((total_samples, 2, n_points), dtype=np.float32) # 噪声残差
    Mask_all = np.ones((total_samples, 1, n_points), dtype=np.float32)     # 有效性掩码

    te_nominal = 0.0010  # 1.0 ms (50 点/回波)
    pts_per_cycle = int(te_nominal * sample_rate)  # 50 点
    num_echoes = int(n_points / pts_per_cycle)

    for i in range(total_samples):
        # 1. 模拟该帧在长衰减序列中的绝对起始时刻 (0 ~ 500 ms)
        t_global_start = np.random.uniform(0.0, 0.45)

        # 2. 从真实 Train 集提取的油脂弛豫参数范围:
        t2_short = np.random.uniform(0.025, 0.055)
        t2_long = np.random.uniform(0.085, 0.240)
        ratio = np.random.uniform(0.35, 0.65)

        # 3. 基础宏观双指数衰减包络 (归一化体系: 初始包络峰值为 1.0)
        t_global = t_global_start + t
        macro_decay = ratio * np.exp(-t_global / t2_short) + (1.0 - ratio) * np.exp(-t_global / t2_long)

        # 4. 周期性自旋回波聚束 (Spin Echo Modulation)
        peak_offset = 0.00076   # 760 us (点 38)
        valley_offset = 0.00027 # 270 us (点 13~14)
        sigma = np.random.uniform(0.00018, 0.00028)

        echo_modulation = np.zeros(n_points, dtype=np.float32)
        for k in range(-1, num_echoes + 2):
            t_center = k * te_nominal + peak_offset
            echo_modulation += np.exp(-0.5 * ((t - t_center) / sigma) ** 2)

        echo_modulation = np.clip(echo_modulation, 0.0, 1.0)
        
        # 纯净信号包络: 回波聚束衰减 + 自然微弱底电平 (约为峰值的 2%~4%)，彻底消除 80% 平底座
        base_leakage = np.random.uniform(0.01, 0.03)
        clean_envelope = macro_decay * (echo_modulation + base_leakage * (1.0 - echo_modulation))

        # 5. 待确认物理凹槽衰减与成对掩码生成
        sample_mask = np.ones(n_points, dtype=np.float32)
        for k in range(-1, num_echoes + 2):
            t_valley = k * te_nominal + valley_offset
            notch_idx = np.where((t >= t_valley - 0.000030) & (t <= t_valley + 0.000030))[0]
            if len(notch_idx) > 0:
                sample_mask[notch_idx] = 0.0
                clean_envelope[notch_idx] *= np.random.uniform(0.05, 0.2)

        # 6. 相位分布 (对齐纽迈真实测量: 虚部为主通道，占 75% 概率)
        phase_angle = np.random.choice([
            np.random.normal(np.pi / 2.0, 0.12),
            np.random.uniform(0, 2.0 * np.pi)
        ], p=[0.75, 0.25])

        clean_i = clean_envelope * np.cos(phase_angle)
        clean_q = clean_envelope * np.sin(phase_angle)

        # 7. 添加真实信噪比范围的混合噪声 (SNR 8 dB ~ 30 dB，对应实际物理数据)
        target_snr_db = np.random.uniform(8.0, 30.0)
        snr_linear = 10.0 ** (target_snr_db / 10.0)
        sig_pow = np.mean(clean_i**2 + clean_q**2) + 1e-12
        noise_std = np.sqrt(sig_pow / snr_linear / 2.0)

        # 仪器直流基线漂移 (< 1.5% 信号峰值)
        dc_i = np.random.normal(0, 0.015 * np.max(clean_envelope))
        dc_q = np.random.normal(0, 0.015 * np.max(clean_envelope))

        noise_i = np.random.normal(0, noise_std, n_points).astype(np.float32)
        noise_q = np.random.normal(0, noise_std, n_points).astype(np.float32)

        noisy_i = clean_i + noise_i + dc_i
        noisy_q = clean_q + noise_q + dc_q

        true_noise_i = noisy_i - clean_i
        true_noise_q = noisy_q - clean_q

        X_all[i, 0, :] = noisy_i
        X_all[i, 1, :] = noisy_q
        Y_clean_all[i, 0, :] = clean_i
        Y_clean_all[i, 1, :] = clean_q
        Y_noise_all[i, 0, :] = true_noise_i
        Y_noise_all[i, 1, :] = true_noise_q
        Mask_all[i, 0, :] = sample_mask

    train_x, train_clean, train_noise, train_mask = (
        X_all[:n_train], Y_clean_all[:n_train], Y_noise_all[:n_train], Mask_all[:n_train]
    )
    val_x, val_clean, val_noise, val_mask = (
        X_all[n_train:n_train + n_val], Y_clean_all[n_train:n_train + n_val],
        Y_noise_all[n_train:n_train + n_val], Mask_all[n_train:n_train + n_val]
    )
    test_x, test_clean, test_noise, test_mask = (
        X_all[n_train + n_val:], Y_clean_all[n_train + n_val:],
        Y_noise_all[n_train + n_val:], Mask_all[n_train + n_val:]
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            train_x=train_x, train_clean=train_clean, train_noise=train_noise, train_mask=train_mask,
            val_x=val_x, val_clean=val_clean, val_noise=val_noise, val_mask=val_mask,
            test_x=test_x, test_clean=test_clean, test_noise=test_noise, test_mask=test_mask,
            train_y=train_clean, val_y=val_clean, test_y=test_clean
        )
        print(f"[阶段 3 重构] 归一化无量纲残差数据集已生成并保存至: {save_path}")
        print(f"  - 训练集: {train_x.shape[0]} 帧 | 验证集: {val_x.shape[0]} 帧 | 测试集: {test_x.shape[0]} 帧")

    return save_path

if __name__ == "__main__":
    _BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.environ.get("LFNMR_M4_NPZ", os.path.join(_BUNDLE_ROOT, "data", "m4_cpmg_multiecho_dataset.npz"))
    generate_cpmg_multiecho_dataset(save_path=OUT)
