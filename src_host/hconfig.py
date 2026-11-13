# -*- coding: utf-8 -*-
"""
hconfig.py —— 上位机大模型 v2 训练总配置

本训练包是对旧 m6_conformer(上位机反演大模型) 的整改性重训，落实 2026-10-23
审阅意见中的三条核心要求:
  1. 统一"生成器与评分谱"的表达(4.4): 谱密度 m 定义在 128 点对数 T2 网格上,
     前向模型 y = A_w @ m 与 NNLS 评分使用同一份谱、同一个前向核;
  2. 杜绝数据重用(4.3 / P3): Conformer 从仿真从头训练, 真实数据严格按固定
     9/3/3 划分 —— 微调只用 Train9, 早停只用 Val3, Test3 仅最终盲测;
  3. 公平对照(P1): 同一份输入上比较 直通NNLS / SG+NNLS / 仿真Conformer /
     微调Conformer 四分支, SG 与 NNLS 正则参数只在开发集上选定后冻结。

所有外部原始数据路径支持通过环境变量只读配置, 全部输出写入当前工作区, 不改动
任何既有文件。
"""

import os

# ---------------------------------------------------------------- 路径配置
# 可移植: 根目录从本文件位置推导(src 的上一级), 拷到任何机器任何路径均可运行。
# (支持跨平台相对路径与环境变量动态推导)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
FIGURE_DIR = os.path.join(ROOT_DIR, "figures")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# 只读引用外部原始数据时请通过环境变量配置; 外部原始波形默认缺失时,
# 真实数据相关脚本会明确跳过或提示, 不影响仿真侧全部功能。
CLOSE_ROOT = os.path.dirname(ROOT_DIR)

# -----------------------------------------------------------------------------
# 双轨路径架构: 支持环境变量覆盖外部只读数据目录，或回退至本包内相对目录
# -----------------------------------------------------------------------------
RAW_DATA_ROOT_EXTERNAL = os.environ.get("LFNMR_RAW_DATA_ROOT", os.path.join(ROOT_DIR, "external_data", "raw_benchtop_cpmg"))
SPLIT_MANIFEST_EXTERNAL = os.environ.get("LFNMR_SPLIT_MANIFEST", os.path.join(ROOT_DIR, "configs_and_metrics", "phase2_fixed_data_split_manifest.csv"))

RAW_DATA_ROOT_LOCAL = os.path.join(CLOSE_ROOT, "06_数据清单", "真实数据_15条核心记录")
SPLIT_MANIFEST_LOCAL = os.path.join(CLOSE_ROOT, "06_数据清单", "phase2_fixed_data_split_manifest.csv")

# 运行时自动调度: 优先使用本地自包含数据，缺失时平滑回退至外部原路径
SPLIT_MANIFEST = SPLIT_MANIFEST_LOCAL if os.path.exists(SPLIT_MANIFEST_LOCAL) else SPLIT_MANIFEST_EXTERNAL
RAW_DATA_ROOT = RAW_DATA_ROOT_LOCAL if os.path.exists(RAW_DATA_ROOT_LOCAL) else RAW_DATA_ROOT_EXTERNAL
REAL_DATA_AVAILABLE = os.path.exists(SPLIT_MANIFEST)

STATE_FILE = os.path.join(ROOT_DIR, "run_state.json")   # 断点续跑状态
MAIN_LOG = os.path.join(LOG_DIR, "train_main.log")

for _d in (MODEL_DIR, RESULT_DIR, FIGURE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- 标准网格
# 与旧 m6_dataset_prep 完全一致, 保证与旧结果/GUI 可比、权重可互换
import numpy as np

CANONICAL_NUM_ECHOES = 1024
CANONICAL_T_MIN_MS = 1.0
CANONICAL_T_MAX_MS = 3000.0

CANONICAL_NUM_T2_BINS = 128
CANONICAL_LOG_T2_MIN = -2.0    # 0.01 ms
CANONICAL_LOG_T2_MAX = 4.0     # 10000 ms

STD_ECHO_TIMES = np.linspace(
    CANONICAL_T_MIN_MS, CANONICAL_T_MAX_MS, CANONICAL_NUM_ECHOES, dtype=np.float64
)
STD_LOG_T2 = np.linspace(
    CANONICAL_LOG_T2_MIN, CANONICAL_LOG_T2_MAX, CANONICAL_NUM_T2_BINS, dtype=np.float64
)
STD_T2_MS = 10.0 ** STD_LOG_T2

# 统一 Fredholm 前向核(含对数网格积分权重):  y_i = sum_j w * exp(-t_i/T2_j) * m_j
# 其中 m_j 是"每单位 log10(T2) 的谱密度", w 是均匀 log 网格间距。
# 该定义同时用于: 仿真生成 / 训练标签 / NNLS 基线 / 全部评分 —— 同一表达, 无再展宽。
LOG_GRID_W = float(STD_LOG_T2[1] - STD_LOG_T2[0])
KERNEL_AW = np.exp(-np.outer(STD_ECHO_TIMES, 1.0 / STD_T2_MS)) * LOG_GRID_W  # (1024,128)


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x 兼容


def trapz_log(spec):
    """谱密度在对数网格上的积分面积"""
    return float(_trapezoid(spec, STD_LOG_T2))


# ---------------------------------------------------------------- 训练预算
# GPU/CPU 自适应: run_all 启动时探测。
# 本机 GPU(RTX 2060+546.33 驱动)对个别 kernel 形状存在偶发 CUDA 非法访存,
# 因此每个 GPU 训练任务在独立子进程中运行, 崩溃自动回退 CPU(预算见下)。
TRAIN_STEPS_GPU = 18000   # 实测~3.5步/s(70%占空比), 18k步约85分钟; 配合早停
CPU_STEPS = 8000          # CPU 回退预算(约 45~70 分钟/trial)
BATCH_SIZE_GPU = 64
BATCH_SIZE_CPU = 32
GPU_DUTY = 0.70           # GPU 占空比上限(用户要求≤70%防黑屏): 每步后按比例补 sleep
VAL_EVERY = 1500          # 每 N 步在固定仿真验证集上评估
EARLY_STOP_PATIENCE = 5   # 连续 N 次评估无提升则早停
TRIAL_TIMEOUT_GPU = 12000  # 单个 GPU 训练子进程超时(秒), 70%占空比下放宽
TRIAL_TIMEOUT_CPU = 10800

# 固定数据集规模与种子(全部显式整数, 不用 hash())
SIM_VAL_N = 1024
SIM_TEST_N = 1024
SIM_VAL_SEED = 101000
SIM_TEST_SEED = 202000
STREAM_SEED = 42

# 真实数据划分(沿用 phase2 固定清单, 只读)
REAL_EPOCHS_MAX = 400
REAL_EARLY_STOP_EVERY = 10
REAL_EARLY_STOP_PATIENCE = 12
