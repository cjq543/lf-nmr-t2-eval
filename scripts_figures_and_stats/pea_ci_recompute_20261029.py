# -*- coding: utf-8 -*-
"""
P-pea 行 CI 复算（审稿循环 R2-M2，作者会话 2026-10-29）

目的：为稿件 Table 4 中 35 条 P-pea 行的 Δr CI（−0.0113 [−0.0205, −0.0021]）
      提供可指认、可复现的存档产出。

方法（与已发布的 35 条 P-mag 四组 CI 完全同一口径，见
12_数字自查_20261027/B2_35条留出四组CI.py）：
  - 记录级配对差（A定稿 − SG旧），35 条；
  - np.random.default_rng(20261025)，rng.integers(0, n, size=(20000, n))，
    逐行取均值，np.percentile([2.5, 97.5]) 百分位法。

步骤：
  1) 先在宽表（configs_and_metrics/holdout35_pmag_wide.csv，P-mag 协议）
     上复现四组已发布 CI，验证代码路径与记录顺序等价（应逐位一致）；
  2) 对长表（configs_and_metrics/holdout35_two_protocol_v3.csv）中 input_protocol=='pea' 的 A定稿/SG旧 逐记录值，
     按宽表的 样品 顺序重排后，用同一口径计算 P-pea 的均值与 CI；
  3) 顺带产出 P-mag 侧同口径对照（长表），与宽表结果互相印证。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(BASE, 'configs_and_metrics', 'holdout35_pmag_wide.csv')
LONG = os.path.join(BASE, 'configs_and_metrics', 'holdout35_two_protocol_v3.csv')
OUTDIR = os.path.join(BASE, 'results')
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, 'pea_ci_recompute_20261029.txt')

lines = []
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)

def boot_ci(d, seed=20261025, n_boot=20000):
    d = pd.Series(d).dropna().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

wide = pd.read_csv(WIDE, encoding='utf-8-sig')
P('宽表:', WIDE)
P('  shape:', wide.shape, ' 样品顺序前5:', list(wide['样品'].head()))

long = pd.read_csv(LONG, encoding='utf-8-sig')
P('长表:', LONG)
P('  shape:', long.shape, ' 协议:', sorted(long['input_protocol'].unique()))

pea = long[long['input_protocol'] == 'pea']
mag = long[long['input_protocol'] == 'raw_mag']

def pivot_protocol(df):
    w = df.pivot_table(index='样品', columns='branch', values='pearson_r', aggfunc='first')
    # 按宽表样品顺序重排（保证与已发布数字同一记录顺序约定）
    order = [s for s in wide['样品'] if s in w.index]
    return w.loc[order]

pea_w = pivot_protocol(pea)
mag_w = pivot_protocol(mag)
P('\npea 透视表 shape:', pea_w.shape, ' 分支:', list(pea_w.columns))
P('mag 透视表 shape:', mag_w.shape)
P('长表/宽表样品集合一致:', set(pea_w.index) == set(wide['样品']))

P('\n=== 第 1 步：宽表(P-mag) 四组已发布 CI 复现（应逐位一致）===')
published = [
    ('A定稿−A旧版',    'A定稿_r', 'A旧版_r',    0.0193, 0.0083, 0.0343),
    ('A定稿−SG旧',     'A定稿_r', 'SG旧_r',     0.0038, -0.0037, 0.0104),
    ('A定稿−SG扩展',   'A定稿_r', 'SG扩展_r',   0.0123, 0.0049, 0.0188),
    ('A定稿−修正NNLS', 'A定稿_r', '修正NNLS_r', 0.0373, 0.0301, 0.0436),
]
for name, a, b, c_m, c_lo, c_hi in published:
    m, lo, hi = boot_ci(wide[a] - wide[b])
    P(f'  {name}: 复算 {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
      f'(发布 {c_m:+.4f} [{c_lo:+.4f}, {c_hi:+.4f}])  '
      f'{"逐位一致" if abs(m-c_m)<5e-5 and abs(lo-c_lo)<5e-5 and abs(hi-c_hi)<5e-5 else "【有偏差】"}')

P('\n=== 第 2 步：P-pea 侧（长表，同口径）===')
for name, a, b in [
    ('A定稿−SG旧（稿件 Table 4 P-pea 行）', 'A定稿', 'SG旧'),
    ('A定稿−A旧版', 'A定稿', 'A旧版'),
    ('A定稿−修正NNLS(直接NNLS)', 'A定稿', '直接NNLS'),
    ('A定稿−SG扩展', 'A定稿', 'SG扩展'),
]:
    m, lo, hi = boot_ci(pea_w[a] - pea_w[b])
    P(f'  P-pea {name}: 均值 {m:+.6f}, 95% CI [{lo:+.4f}, {hi:+.4f}]')
    P(f'    分支均值: A定稿={pea_w[a].mean():.6f}, {b}={pea_w[b].mean():.6f}, 胜场={int(((pea_w[a]-pea_w[b])>0).sum())}/35')

P('\n=== 第 3 步：P-mag 侧（长表，交叉印证）===')
for name, a, b in [('A定稿−SG旧', 'A定稿', 'SG旧')]:
    m, lo, hi = boot_ci(mag_w[a] - mag_w[b])
    P(f'  P-mag {name}: 均值 {m:+.6f}, 95% CI [{lo:+.4f}, {hi:+.4f}]')
    P(f'    分支均值: A定稿={mag_w[a].mean():.6f}, {b}={mag_w[b].mean():.6f}')

P('\n结论要点见 results_新增/pea_ci_recompute_20261029.txt 与本脚本。')

os.makedirs(OUTDIR, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n结果已写入', OUT)
