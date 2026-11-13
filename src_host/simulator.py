# -*- coding: utf-8 -*-
"""
simulator.py —— 统一表达的高保真 CPMG 弛豫谱仿真生成器

统一性约定(整改审阅意见 4.4):
  * 谱密度 m 定义在 STD_LOG_T2 (128 点, 0.01~10000 ms 均匀对数网格) 上;
  * 前向包络 y = KERNEL_AW @ m (核内已含 log 网格间距权重);
  * 训练标签、NNLS 基线、全部谱相关度评分都用这同一份 m / 同一个前向,
    不存在"生成用离散组分、评分另用高斯展宽"的模型不自洽问题。

噪声模型(包络级, 对应 .pea 回波峰值重采样到 1024 点后的表示):
  * 加性高斯白噪, SNR 对数均匀 15~45 dB, 另以 10% 概率抽 8~15 dB 低信噪比;
  * 微小直流基线偏移 + 慢漂移;
  * 全部随机性来自显式整数种子的 default_rng, 跨进程可复现。
"""

import numpy as np

from hconfig import STD_LOG_T2, STD_T2_MS, STD_ECHO_TIMES, KERNEL_AW, trapz_log

# 食用油物理先验的峰参数范围 (log10 空间)
# 主峰: 甘油三酯本体 80~900 ms; 次峰: 氧化裂解短链 10~80 ms;
# 微弱短峰: 结合相/聚合物 1~10 ms; 微弱长峰: 3%~8% 占比的 500~2000 ms
# (审阅意见 P2 明确要求补充"真实长 T2 / 弱组分"案例)
_PEAK_RANGES = [
    ("main",  np.log10(80.0),   np.log10(900.0),  0.15, 0.40),
    ("sub",   np.log10(10.0),   np.log10(80.0),   0.12, 0.35),
    ("short", np.log10(1.0),    np.log10(10.0),   0.10, 0.30),
    ("long",  np.log10(500.0),  np.log10(2000.0), 0.10, 0.25),
]


def _sample_spectrum(rng):
    """随机生成 1~4 峰对数高斯谱密度, 返回 (m, peaks)。
    peaks: [(t2_ms, area_fraction), ...] 供组分恢复指标使用。"""
    n_main_like = rng.choice([1, 2, 3], p=[0.25, 0.55, 0.20])
    spec = np.zeros(STD_LOG_T2.size, dtype=np.float64)
    peaks = []

    # 必有主峰
    mu = rng.uniform(np.log10(80.0), np.log10(900.0))
    sig = rng.uniform(0.15, 0.40)
    amp = rng.uniform(0.5, 1.0)
    spec += amp * np.exp(-0.5 * ((STD_LOG_T2 - mu) / sig) ** 2)
    raw_amp = [amp]

    # 次峰 / 短峰 (氧化变质、结合相)
    if n_main_like >= 2:
        mu2 = rng.uniform(np.log10(10.0), np.log10(80.0))
        sig2 = rng.uniform(0.12, 0.35)
        amp2 = rng.uniform(0.10, 0.60) * amp
        spec += amp2 * np.exp(-0.5 * ((STD_LOG_T2 - mu2) / sig2) ** 2)
        raw_amp.append(amp2)
    if n_main_like >= 3:
        mu3 = rng.uniform(np.log10(1.0), np.log10(10.0))
        sig3 = rng.uniform(0.10, 0.30)
        amp3 = rng.uniform(0.05, 0.30) * amp
        spec += amp3 * np.exp(-0.5 * ((STD_LOG_T2 - mu3) / sig3) ** 2)
        raw_amp.append(amp3)

    # 以 35% 概率额外出现微弱长 T2 组分(弱组分压力案例)
    has_long = bool(rng.random() < 0.35)
    if has_long:
        mu4 = rng.uniform(np.log10(500.0), np.log10(2000.0))
        sig4 = rng.uniform(0.10, 0.25)
        amp4 = rng.uniform(0.03, 0.10) * amp
        spec += amp4 * np.exp(-0.5 * ((STD_LOG_T2 - mu4) / sig4) ** 2)
        raw_amp.append(amp4)

    # 由谱本身数值积分得到各峰面积分数(与前向核同一离散化, 无表达差异)
    w = STD_LOG_T2[1] - STD_LOG_T2[0]
    masses = []
    for a, mu_i, sig_i in zip(
        raw_amp,
        [mu] + ([mu2] if n_main_like >= 2 else []) + ([mu3] if n_main_like >= 3 else [])
               + ([mu4] if has_long else []),
        [sig] + ([sig2] if n_main_like >= 2 else []) + ([sig3] if n_main_like >= 3 else [])
               + ([sig4] if has_long else []),
    ):
        peak = a * np.exp(-0.5 * ((STD_LOG_T2 - mu_i) / sig_i) ** 2)
        masses.append(float(np.sum(peak) * w))
    total = sum(masses)
    peaks = [(float(STD_T2_MS[int(np.argmin(np.abs(STD_LOG_T2 - m_i)))]), fr)
             for m_i, fr in zip(
                 [mu] + ([mu2] if n_main_like >= 2 else []) + ([mu3] if n_main_like >= 3 else [])
                        + ([mu4] if has_long else []),
                 [m_ / total for m_ in masses])]

    return spec, peaks


def _sample_noise(rng, n):
    """包络级噪声: SNR 对数均匀 15~45 dB, 10% 概率 8~15 dB; 加基线偏移与漂移。"""
    if rng.random() < 0.10:
        snr_db = rng.uniform(8.0, 15.0)
    else:
        snr_db = np.exp(rng.uniform(np.log(15.0), np.log(45.0)))
    sigma = 10.0 ** (-snr_db / 20.0)
    noise = rng.normal(0.0, sigma, size=n)
    baseline = rng.normal(0.0, 0.004) + rng.normal(0.0, 0.002) * (np.arange(n) / n)
    return noise + baseline, snr_db


def generate_sample(seed):
    """生成单个样本: (noisy_echo 1024, label_spec 128, peaks, snr_db)。
    归一化约定: 输入与标签同除以干净包络首点 y[0], 因此输入首点≈1,
    标签谱的 log 网格积分面积≈1。"""
    rng = np.random.default_rng(seed)
    spec, peaks = _sample_spectrum(rng)
    y_clean = KERNEL_AW @ spec
    scale = float(y_clean[0])
    if scale < 1e-9:
        scale = 1.0
    noise, snr_db = _sample_noise(rng, y_clean.size)
    noisy = (y_clean / scale) + noise
    label = spec / scale
    return noisy.astype(np.float32), label.astype(np.float32), peaks, float(snr_db)


def generate_dataset(n, base_seed, snr_filter=None):
    """批量生成固定数据集(验证/测试用)。snr_filter: None 或 'low' 只保留 <15dB。"""
    xs, ys, metas = [], [], []
    seed = base_seed
    while len(xs) < n:
        x, y, peaks, snr = generate_sample(seed)
        seed += 1
        if snr_filter == "low" and snr >= 15.0:
            continue
        xs.append(x)
        ys.append(y)
        metas.append({"seed": seed - 1, "snr_db": snr, "peaks": peaks})
    return (np.stack(xs), np.stack(ys), metas)


class StreamSimDataset:
    """流式无限训练集: 每个batch用 (STREAM_SEED, step) 派生的稳定种子现生成,
    保证同一 step 跨进程可复现, 且训练永远见得到新鲜样本(不重复采样)。"""

    def __init__(self, stream_seed, batch_size):
        self.stream_seed = stream_seed
        self.batch_size = batch_size

    def batch(self, step):
        seeds = [self.stream_seed * 1000003 + step * 8191 + k for k in range(self.batch_size)]
        xs, ys = [], []
        for s in seeds:
            x, y, _, _ = generate_sample(s)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)


# =====================================================================
# 定稿版(v3)扩展分布 —— 依据 OOD 实验暴露的适用范围边界设计:
#   V3 峰位外推失败 -> 主峰 T2 范围 80~900 扩至 30~2500 ms
#   V4 多组分失败   -> 峰数 1~3(+长) 扩至 1~6
#   V2 宽峰略输     -> 峰宽上限 0.40 -> 0.55
#   V6 截断崩溃     -> 30% 概率随机截断输入(400~2800ms), 标签保持完整谱;
#                     截断样本返回有效窗 mask, 物理损失/前向指标只算有效区
#   V5/V7 已胜      -> 噪声与基线模型保持不变
# 旧分布行为完全保留(extended=False 路径未动), 历史结果仍可复现。
# =====================================================================

def _sample_spectrum_ext(rng):
    """扩展分布随机谱: 主峰 log-U(30,2500)ms σU(0.10,0.55), 附加 0~5 个
    log-U(2,2500)ms 组分, 35% 概率微弱长 T₂ 组分(与原分布一致)。"""
    k = int(rng.choice([1, 2, 3, 4, 5, 6],
                       p=[0.15, 0.35, 0.25, 0.12, 0.08, 0.05]))
    raw = []
    main_mu = rng.uniform(np.log10(30.0), np.log10(2500.0))
    main_sig = rng.uniform(0.10, 0.55)
    main_amp = rng.uniform(0.5, 1.0)
    raw.append((main_mu, main_sig, main_amp))
    for _ in range(k - 1):
        raw.append((rng.uniform(np.log10(2.0), np.log10(2500.0)),
                    rng.uniform(0.08, 0.35),
                    rng.uniform(0.03, 0.6) * main_amp))
    if rng.random() < 0.35:
        raw.append((rng.uniform(np.log10(500.0), np.log10(2500.0)),
                    rng.uniform(0.10, 0.25),
                    rng.uniform(0.03, 0.10) * main_amp))

    spec = np.zeros(STD_LOG_T2.size, dtype=np.float64)
    for mu, sig, amp in raw:
        spec += amp * np.exp(-0.5 * ((STD_LOG_T2 - mu) / sig) ** 2)

    w = STD_LOG_T2[1] - STD_LOG_T2[0]
    masses = [float(np.sum(a * np.exp(-0.5 * ((STD_LOG_T2 - m) / s) ** 2)) * w)
              for m, s, a in raw]
    total = sum(masses)
    peaks = [(float(STD_T2_MS[int(np.argmin(np.abs(STD_LOG_T2 - m)))]), ms / total)
             for m, ms in zip([r[0] for r in raw], masses)]
    return spec, peaks


def generate_sample_ext(seed):
    """扩展分布单样本: 返回 (x, y, valid_mask, snr_db)。
    30% 概率在 400~2800ms 截断输入(mask 同步置零), 标签谱保持完整。"""
    rng = np.random.default_rng(seed)
    spec, peaks = _sample_spectrum_ext(rng)
    y = KERNEL_AW @ spec
    scale = float(y[0]) if y[0] > 1e-9 else 1.0
    noise, snr = _sample_noise(rng, y.size)
    x = y / scale + noise
    mask = np.ones(y.size, dtype=np.float32)
    if rng.random() < 0.30:
        cut_t = rng.uniform(400.0, 2800.0)
        cut = max(int(np.searchsorted(STD_ECHO_TIMES, cut_t)), 64)
        x[cut:] = 0.0
        mask[cut:] = 0.0
    return (x.astype(np.float32), (spec / scale).astype(np.float32), mask, snr)


def generate_dataset_ext(n, base_seed):
    """扩展分布固定数据集(定稿版验证/测试用)。"""
    xs, ys, ms = [], [], []
    for k in range(n):
        x, y, m, _ = generate_sample_ext(base_seed + k)
        xs.append(x); ys.append(y); ms.append(m)
    return np.stack(xs), np.stack(ys), np.stack(ms)


class ExtendedStreamDataset:
    """扩展分布流式训练集: batch 返回 (x, y, valid_mask) 三元组。"""

    def __init__(self, stream_seed, batch_size):
        self.stream_seed = stream_seed
        self.batch_size = batch_size

    def batch(self, step):
        seeds = [self.stream_seed * 1000003 + step * 8191 + k
                 for k in range(self.batch_size)]
        xs, ys, ms = [], [], []
        for s in seeds:
            x, y, m, _ = generate_sample_ext(s)
            xs.append(x); ys.append(y); ms.append(m)
        return np.stack(xs), np.stack(ys), np.stack(ms)
