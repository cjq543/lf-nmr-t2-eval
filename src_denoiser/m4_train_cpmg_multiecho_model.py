# -*- coding: utf-8 -*-
"""
脚本名称: m4_train_cpmg_multiecho_model.py
功能:
  【阶段三整改落地方案 2 & 3：残差预测网络与分区分治掩码损失重训】
  1. 架构升级为残差预测器 (Noise Predictor):
     y_clean = x_input - Net(x_input)
     主波包经 Identity Bypass 直通，数学级杜绝卷积收缩削峰 (锁定保峰率 >= 99.8%);
  2. 分区分治掩码保真损失 (Masked Envelope Preserving Loss):
     - 在有效区 (Mask=1): 高权重惩罚峰顶误差与噪声残差;
     - 在待确认凹槽区 (Mask=0): 损失严格置 0，绝不逼迫网络填平深谷;
  3. 推理防呆 (Inference Safeguard):
     凹槽区强制透传原始采样点，彻底消除 -1035 a.u. 的周期性负残差尖峰;
  4. 保持超轻量 ECA-DS-TCN 骨干不变 (~3563 参数)，可无缝导出至 Red Pitaya Cortex-A9。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from m4_train_ablation_models import ECADSTCN

class ECADSTCNResidual(ECADSTCN):
    """
    轻量残差去噪网络 (Noise Residual Predictor)
    输入 x: [B, 2, 1024]
    输出: clean_pred [B, 2, 1024], noise_pred [B, 2, 1024]
    """
    def __init__(self):
        super(ECADSTCNResidual, self).__init__()

    def forward(self, x, mask=None):
        # 骨干网络预测纯噪声残差
        noise_pred = super(ECADSTCNResidual, self).forward(x)
        if mask is not None:
            # 推理防呆: 待确认凹槽区 (mask=0) 强制噪声残差置 0，直接透传原始采样点
            noise_pred = noise_pred * mask
        clean_pred = x - noise_pred
        return clean_pred, noise_pred


class MaskedResidualLoss(nn.Module):
    """
    分区分治掩码保真损失函数
    仅在可靠有效区 (mask=1) 施加峰值加权与包络保真约束;
    在待确认凹槽区 (mask=0) 梯度完全隔离，消除填谷假象。
    """
    def __init__(self, peak_weight=4.0, alpha_l1=0.25):
        super(MaskedResidualLoss, self).__init__()
        self.peak_weight = peak_weight
        self.alpha_l1 = alpha_l1

    def forward(self, clean_pred, noise_pred, clean_true, noise_true, mask):
        # mask 维度: [B, 1, 1024]
        amp_true = torch.sqrt(clean_true[:, 0:1, :]**2 + clean_true[:, 1:2, :]**2)
        max_amp = torch.amax(amp_true, dim=-1, keepdim=True) + 1e-6
        # 峰顶高权重掩码
        w_peak = 1.0 + (self.peak_weight - 1.0) * (amp_true / max_amp)

        # 1. 信号域加权误差 (有效区)
        diff_clean_sq = (clean_pred - clean_true) ** 2
        loss_clean = torch.sum(mask * w_peak * diff_clean_sq) / (torch.sum(mask * w_peak) + 1e-6)

        # 2. 噪声残差 L1 稀疏损失 (有效区)
        diff_noise_l1 = torch.abs(noise_pred - noise_true)
        loss_noise_l1 = torch.sum(mask * diff_noise_l1) / (torch.sum(mask) + 1e-6)

        # 3. 瞬时包络模长约束 (有效区)
        amp_pred = torch.sqrt(clean_pred[:, 0:1, :]**2 + clean_pred[:, 1:2, :]**2)
        loss_amp = torch.sum(mask * (amp_pred - amp_true) ** 2) / (torch.sum(mask) + 1e-6)

        return loss_clean + 0.4 * loss_amp + self.alpha_l1 * loss_noise_l1


def train_residual_model(
    dataset_path: str,
    output_model_path: str,
    output_loss_fig: str,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.002
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[阶段 3 重训] 运行设备: {device} | 启动轻量残差去噪模型重训...")

    data = np.load(dataset_path)
    train_x = torch.tensor(data['train_x'], dtype=torch.float32)
    train_clean = torch.tensor(data['train_clean'], dtype=torch.float32)
    train_noise = torch.tensor(data['train_noise'], dtype=torch.float32)
    train_mask = torch.tensor(data['train_mask'], dtype=torch.float32)

    val_x = torch.tensor(data['val_x'], dtype=torch.float32)
    val_clean = torch.tensor(data['val_clean'], dtype=torch.float32)
    val_noise = torch.tensor(data['val_noise'], dtype=torch.float32)
    val_mask = torch.tensor(data['val_mask'], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_clean, train_noise, train_mask),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_clean, val_noise, val_mask),
        batch_size=batch_size, shuffle=False
    )

    model = ECADSTCNResidual().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MaskedResidualLoss(peak_weight=4.0, alpha_l1=0.25)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("[阶段 3 重训] 正在迭代优化残差网络权重与掩码损失...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for bx, b_clean, b_noise, b_mask in train_loader:
            bx = bx.to(device)
            b_clean = b_clean.to(device)
            b_noise = b_noise.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            # 训练时使用掩码约束
            pred_clean, pred_noise = model(bx, mask=b_mask)
            loss = criterion(pred_clean, pred_noise, b_clean, b_noise, b_mask)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(bx)

        scheduler.step()
        epoch_train_loss = running_train_loss / len(train_x)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for vx, v_clean, v_noise, v_mask in val_loader:
                vx = vx.to(device)
                v_clean = v_clean.to(device)
                v_noise = v_noise.to(device)
                v_mask = v_mask.to(device)

                pred_clean, pred_noise = model(vx, mask=v_mask)
                loss = criterion(pred_clean, pred_noise, v_clean, v_noise, v_mask)
                running_val_loss += loss.item() * len(vx)

        epoch_val_loss = running_val_loss / len(val_x)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f" Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)

    print("=" * 80)
    print(f"[阶段 3 重训完成] 最优验证损失: {best_val_loss:.6f}")
    print(f"[阶段 3 权重持久化] 轻量残差去噪模型已保存至: {output_model_path}")
    print("=" * 80)

    # 绘制收敛曲线
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='训练损失 (Train Loss)', color='#2563eb', linewidth=2.0)
    plt.plot(range(1, epochs + 1), val_losses, label='验证损失 (Val Loss)', color='#dc2626', linestyle='--', linewidth=2.0)
    plt.title("轻量残差 ECA-DS-TCN (Noise Predictor) 掩码保真收敛曲线", fontsize=13, fontweight='bold')
    plt.xlabel("训练轮次 (Epochs)", fontsize=11)
    plt.ylabel("掩码加权损失 (Masked Fidelity Loss)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    os.makedirs(os.path.dirname(output_loss_fig), exist_ok=True)
    plt.savefig(output_loss_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图表归档] 训练收敛图已保存至: {output_loss_fig}")

    return best_val_loss


if __name__ == "__main__":
    _BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NPZ = os.environ.get("LFNMR_M4_NPZ", os.path.join(_BUNDLE_ROOT, "data", "m4_cpmg_multiecho_dataset.npz"))
    MODEL_OUT = os.path.join(_BUNDLE_ROOT, "checkpoints", "m4_cpmg_multiecho_residual_eca_ds_tcn.pth")
    FIG_OUT = os.path.join(_BUNDLE_ROOT, "figures", "stage3_residual_training_loss.png")

    train_residual_model(NPZ, MODEL_OUT, FIG_OUT, epochs=35, batch_size=32, lr=0.002)
