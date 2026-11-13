# -*- coding: utf-8 -*-
"""
models.py —— 上位机 1D-Conformer-ResUNet 反演大模型(独立副本)

架构与旧 m6_conformer_resunet.ConformerResUNet1D 完全一致(类名/参数名/前向逻辑
逐一对应), 因此:
  * state_dict 键完全兼容, 训练出的权重可直接被现有 MaRGE/演示 GUI 加载;
  * 与旧 m6_conformer_best_model.pth 的结果可严格对比。

物理引导损失同样保留(Huber + Wasserstein + Fredholm 前向自洽 + 稀疏), 但前向核
改用统一加权的 KERNEL_AW, 消除旧实现中生成与评分表达不一致的问题。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hconfig import KERNEL_AW

# 稳定性规避(本机实测): RTX 2060 (Turing, sm_75) + 驱动 546.33 上,
# torch 默认 SDPA 后端(flash / mem-efficient)的反向 kernel 在 nn.MultiheadAttention
# 训练中触发 CUDA illegal memory access(torch 2.5.1+cu121 与 2.14.1+cu126 均复现,
# conv/batchnorm/matmul 各 400 次迭代均正常, 仅注意力算子故障; 系统日志伴随 nvlddmkm
# GPU 异常事件)。强制 MATH 后端后 1500 次训练迭代完全稳定, 且本模型注意力仅在
# 长度 128 的瓶颈段使用, 速度损失可忽略。仅在使用 CUDA 时切换。
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # 降频后仍出现 cuDNN CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH(约1800步处),
    # 关闭 cuDNN 走原生卷积路径; 本模型全部为小尺寸 1D 卷积, 速度影响很小
    torch.backends.cudnn.enabled = False


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class ConformerConvModule1D(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # 属性名与旧 m6_conformer_resunet 逐字一致, 保证 state_dict 键完全兼容
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size,
                                        padding=kernel_size // 2, groups=d_model,
                                        bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B, L, D)
        residual = x
        x = self.ln(x).transpose(1, 2)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.act(self.bn(self.depthwise_conv(x)))
        x = self.drop(self.pointwise_conv2(x)).transpose(1, 2)
        return residual + x


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.drop(self.fc2(self.act(self.fc1(x))))
        return residual + 0.5 * x


class ConformerBlock1D(nn.Module):
    def __init__(self, d_model, num_heads=4, conv_kernel=15, dropout=0.1):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, 4, dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule1D(d_model, conv_kernel, dropout)
        self.ffn2 = ConformerFeedForward(d_model, 4, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ffn1(x)
        n = self.attn_ln(x)
        a, _ = self.attn(n, n, n, need_weights=False)
        x = x + self.attn_drop(a)
        x = self.conv_module(x)
        x = self.ffn2(x)
        return self.final_ln(x)


class ConformerResUNet1D(nn.Module):
    """输入 (B,1,1024) 归一化回波包络 -> 输出 (B,128) 非负 T2 谱密度"""

    def __init__(self, in_channels=1, out_bins=128, stem_dim=32, d_model=128,
                 num_heads=4, num_conformer_layers=3, dropout=0.05):
        super().__init__()
        self.out_bins = out_bins
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_dim),
            nn.GELU(),
        )
        self.enc1 = ResBlock1D(stem_dim, 64, stride=2)
        self.enc2 = ResBlock1D(64, d_model, stride=2)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock1D(d_model, num_heads, 15, dropout)
            for _ in range(num_conformer_layers)
        ])
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(d_model, 64, 3, padding=1),
        )
        self.dec2 = ResBlock1D(64 + 64, 64)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(64, stem_dim, 3, padding=1),
        )
        self.dec1 = ResBlock1D(stem_dim + stem_dim, stem_dim)
        self.head_pool = nn.AdaptiveAvgPool1d(out_bins)
        self.head_proj = nn.Sequential(
            nn.Linear(stem_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                       # (B,1,1024)
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(e1)
        h = e2.transpose(1, 2)
        for blk in self.conformer_blocks:
            h = blk(h)
        h = h.transpose(1, 2)
        d2 = self.up2(h)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s0], dim=1))
        p = self.head_pool(d1).transpose(1, 2)  # (B,128,32)
        out = self.head_proj(p).squeeze(-1)     # (B,128)
        return F.softplus(out)


class UnifiedInversionLoss(nn.Module):
    """物理引导复合损失(统一加权前向核版)"""

    def __init__(self, lambda_huber=1.0, lambda_wass=0.5, lambda_phys=0.3,
                 lambda_sparse=0.01, huber_beta=0.01):
        super().__init__()
        self.lambda_huber = lambda_huber
        self.lambda_wass = lambda_wass
        self.lambda_phys = lambda_phys
        self.lambda_sparse = lambda_sparse
        self.huber = nn.SmoothL1Loss(beta=huber_beta)
        self.register_buffer("kernel_AW",
                             torch.tensor(KERNEL_AW, dtype=torch.float32))

    @staticmethod
    def wasserstein_1d(pred, target, eps=1e-8):
        p = pred / (pred.sum(dim=-1, keepdim=True) + eps)
        t = target / (target.sum(dim=-1, keepdim=True) + eps)
        return torch.mean(torch.abs(torch.cumsum(p, -1) - torch.cumsum(t, -1)))

    def forward(self, pred_spec, true_spec, input_echo, valid_mask=None):
        l_huber = self.huber(pred_spec, true_spec)
        l_wass = self.wasserstein_1d(pred_spec, true_spec)
        y_recon = torch.matmul(pred_spec, self.kernel_AW.t())
        echo = input_echo.squeeze(1)
        if valid_mask is None:
            l_phys = F.mse_loss(y_recon, echo)
        else:
            # 截断样本: 前向自洽损失只在有效时间窗内计算(定稿v3协议)
            diff = (y_recon - echo) * valid_mask
            l_phys = diff.pow(2).sum() / (valid_mask.sum() + 1e-8)
        l_sparse = torch.mean(pred_spec)
        total = (self.lambda_huber * l_huber + self.lambda_wass * l_wass
                 + self.lambda_phys * l_phys + self.lambda_sparse * l_sparse)
        return total, {
            "loss_total": float(total.detach()),
            "loss_huber": float(l_huber.detach()),
            "loss_wass": float(l_wass.detach()),
            "loss_phys": float(l_phys.detach()),
            "loss_sparse": float(l_sparse.detach()),
        }
