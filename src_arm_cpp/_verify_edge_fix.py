# -*- coding: utf-8 -*-
"""_verify_edge_fix.py —— 验证修复后的 C++ 边缘输出与 scipy 参考逐位一致"""
import os
import numpy as np

base = os.environ.get('LFNMR_SG21_DIR', os.path.dirname(os.path.abspath(__file__)))
fx = np.fromfile(os.path.join(base, '_probe_fix_output.iq.bin'), dtype=np.float32).reshape(-1, 2).T
p = np.fromfile(os.path.join(base, 'pc_ref', '大豆油24h.sg21pc.iq.bin'), dtype=np.float32).reshape(-1, 2).T
half = 10
print('I 左边缘前 5（修复后 C++ vs scipy）:')
for i in range(5):
    print(f'  [{i}] fix={fx[0][i]:+.6f}  scipy={p[0][i]:+.6f}')
for ch, cn in ((0, 'I'), (1, 'Q')):
    d_in = np.abs(fx[ch][half:-half].astype(np.float64) - p[ch][half:-half].astype(np.float64))
    d_ed = np.abs(np.concatenate([fx[ch][:half], fx[ch][-half:]]).astype(np.float64)
                  - np.concatenate([p[ch][:half], p[ch][-half:]]).astype(np.float64))
    r_all = np.corrcoef(fx[ch].astype(np.float64), p[ch].astype(np.float64))[0, 1]
    rms = float(np.sqrt(np.mean(p[ch].astype(np.float64) ** 2)))
    print(f'{cn}: 内部 relRMSE={np.sqrt(np.mean(d_in**2))/rms:.3e}  '
          f'边缘 relRMSE={np.sqrt(np.mean(d_ed**2))/rms:.3e}  '
          f'边缘 maxabs={d_ed.max():.3e}  整体 r={r_all:.9f}')
