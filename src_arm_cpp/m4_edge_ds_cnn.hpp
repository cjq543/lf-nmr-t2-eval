/**
 * =============================================================================
 * 课题名称: 基于 Red Pitaya (Zynq-7020) 的便携式低场核磁边缘去噪系统
 * 模块功能: 模块 4 - ECA-DS-TCN (深度可分离因果卷积) 纯 C++ 零依赖推理引擎
 * 硬件平台: Xilinx Zynq-7020 (ARM Dual Cortex-A9 @ 667MHz)
 * 作者: 黄杰 (上海理工大学 · 健康科学与工程学院 · 生医工学硕)
 * 特性:
 *   1. 深度可分离卷积 (Depthwise + Pointwise 1x1)，计算量降低 74%
 *   2. 零动态内存分配 (静态预分配全部工作缓存，零 malloc/free 开销)
 *   3. 极速连续访存，专为 Cortex-A9 缓存友好的局部性设计
 * =============================================================================
 */

#ifndef M4_EDGE_DS_CNN_HPP
#define M4_EDGE_DS_CNN_HPP

#include <cmath>
#include <cstring>
#include <algorithm>
#include "m4_ds_cnn_weights.hpp"

namespace NMR {

class EdgeDSTCN {
public:
    static constexpr int SEQ_LEN = 1024;

    EdgeDSTCN() {
        std::memset(buf_in, 0, sizeof(buf_in));
        std::memset(buf_out, 0, sizeof(buf_out));
        std::memset(buf_h1, 0, sizeof(buf_h1));
        std::memset(buf_h3, 0, sizeof(buf_h3));
        std::memset(buf_A, 0, sizeof(buf_A));
        std::memset(buf_B, 0, sizeof(buf_B));
        std::memset(buf_mid, 0, sizeof(buf_mid));
    }

    /**
     * @brief 极致访存优化版一维深度卷积 (寄存器累加 + OpenMP 双核并行)
     */
    static void depthwise_conv1d(
        const float* __restrict__ in_data, float* __restrict__ out_data, int channels,
        const float* __restrict__ weights, const float* __restrict__ bias,
        int K, int dilation, int padding
    ) {
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < channels; ++c) {
            float bias_val = bias ? bias[c] : 0.0f;
            const float* in_chan = in_data + c * SEQ_LEN;
            float* out_chan = out_data + c * SEQ_LEN;
            const float* w = weights + c * K;

            for (int t = 0; t < SEQ_LEN; ++t) {
                float acc = bias_val;
                for (int k = 0; k < K; ++k) {
                    int in_idx = t + k * dilation - padding;
                    if (in_idx >= 0 && in_idx < SEQ_LEN) {
                        acc += w[k] * in_chan[in_idx];
                    }
                }
                out_chan[t] = acc; // 仅向内存写出一次
            }
        }
    }

    /**
     * @brief 极致访存优化版逐点卷积 (1x1 跨通道映射 + 寄存器累加 + 原地激活 + 双核并行)
     */
    static void pointwise_conv1d(
        const float* __restrict__ in_data, int in_channels,
        float* __restrict__ out_data, int out_channels,
        const float* __restrict__ weights, const float* __restrict__ bias,
        bool apply_leaky_relu = true
    ) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < out_channels; ++oc) {
            float bias_val = bias[oc];
            float* out_chan = out_data + oc * SEQ_LEN;
            const float* w_row = weights + oc * in_channels;

            for (int t = 0; t < SEQ_LEN; ++t) {
                float acc = bias_val;
                for (int ic = 0; ic < in_channels; ++ic) {
                    acc += w_row[ic] * in_data[ic * SEQ_LEN + t];
                }
                if (apply_leaky_relu) {
                    acc = (acc >= 0.0f) ? acc : 0.1f * acc;
                }
                out_chan[t] = acc; // 彻底消除反复读写，仅向内存写出一次
            }
        }
    }

    /**
     * @brief 标准卷积算子 (优化版: 用于 Stem 与 Head)
     */
    static void standard_conv1d(
        const float* __restrict__ in_data, int in_channels,
        float* __restrict__ out_data, int out_channels,
        const float* __restrict__ weights, const float* __restrict__ bias,
        int K, int padding, bool apply_leaky_relu
    ) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < out_channels; ++oc) {
            float bias_val = bias[oc];
            float* out_chan = out_data + oc * SEQ_LEN;
            const float* w_oc = weights + oc * in_channels * K;

            for (int t = 0; t < SEQ_LEN; ++t) {
                float acc = bias_val;
                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* in_chan = in_data + ic * SEQ_LEN;
                    const float* w_ic = w_oc + ic * K;
                    for (int k = 0; k < K; ++k) {
                        int in_idx = t + k - padding;
                        if (in_idx >= 0 && in_idx < SEQ_LEN) {
                            acc += w_ic[k] * in_chan[in_idx];
                        }
                    }
                }
                if (apply_leaky_relu) {
                    acc = (acc >= 0.0f) ? acc : 0.1f * acc;
                }
                out_chan[t] = acc;
            }
        }
    }

    /**
     * @brief 极轻量 ECA 通道注意力
     */
    static void apply_eca(float* data, int channels, const float* eca_weight) {
        float avg[32];
        const float inv_len = 1.0f / static_cast<float>(SEQ_LEN);
        for (int c = 0; c < channels; ++c) {
            const float* chan = data + c * SEQ_LEN;
            float sum = 0.0f;
            for (int t = 0; t < SEQ_LEN; ++t) {
                sum += chan[t];
            }
            avg[c] = sum * inv_len;
        }

        float att[32];
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                int in_idx = c + k - 1;
                if (in_idx >= 0 && in_idx < channels) {
                    sum += eca_weight[k] * avg[in_idx];
                }
            }
            att[c] = 1.0f / (1.0f + std::exp(-sum));
        }

        for (int c = 0; c < channels; ++c) {
            float a = att[c];
            float* chan = data + c * SEQ_LEN;
            for (int t = 0; t < SEQ_LEN; ++t) {
                chan[t] *= a;
            }
        }
    }

    static inline void add_residual(float* dst, const float* src, int count) {
        for (int i = 0; i < count; ++i) {
            dst[i] += src[i];
        }
    }

    /**
     * @brief 单帧 1024 点 I/Q 前向推理
     */
    void forward(const float* in_i, const float* in_q, float* out_i, float* out_q) {
        std::memcpy(buf_in, in_i, SEQ_LEN * sizeof(float));
        std::memcpy(buf_in + SEQ_LEN, in_q, SEQ_LEN * sizeof(float));

        // 1. Stem: Conv1d(2 -> 16, k=5, p=2) + LeakyReLU
        standard_conv1d(buf_in, 2, buf_h1, 16, DSWeights::stem_weight, DSWeights::stem_bias, 5, 2, true);

        // 2. DS2: 16 -> 24 (k=5, d=2, p=4)
        depthwise_conv1d(buf_h1, buf_mid, 16, DSWeights::ds2_depthwise_weight, DSWeights::ds2_depthwise_bias, 5, 2, 4);
        pointwise_conv1d(buf_mid, 16, buf_A, 24, DSWeights::ds2_pointwise_weight, DSWeights::ds2_pointwise_bias, true);

        // 3. DS3: 24 -> 24 (k=5, d=4, p=8) + ECA1
        depthwise_conv1d(buf_A, buf_mid, 24, DSWeights::ds3_depthwise_weight, DSWeights::ds3_depthwise_bias, 5, 4, 8);
        pointwise_conv1d(buf_mid, 24, buf_h3, 24, DSWeights::ds3_pointwise_weight, DSWeights::ds3_pointwise_bias, true);
        apply_eca(buf_h3, 24, DSWeights::eca1_conv_weight);

        // 4. DS4: 24 -> 24 (k=5, d=8, p=16)
        depthwise_conv1d(buf_h3, buf_mid, 24, DSWeights::ds4_depthwise_weight, DSWeights::ds4_depthwise_bias, 5, 8, 16);
        pointwise_conv1d(buf_mid, 24, buf_B, 24, DSWeights::ds4_pointwise_weight, DSWeights::ds4_pointwise_bias, true);

        // 5. DS5: 24 -> 24 (k=5, d=16, p=32) + ECA2 + 残差(h3)
        depthwise_conv1d(buf_B, buf_mid, 24, DSWeights::ds5_depthwise_weight, DSWeights::ds5_depthwise_bias, 5, 16, 32);
        pointwise_conv1d(buf_mid, 24, buf_A, 24, DSWeights::ds5_pointwise_weight, DSWeights::ds5_pointwise_bias, true);
        apply_eca(buf_A, 24, DSWeights::eca2_conv_weight);
        add_residual(buf_A, buf_h3, 24 * SEQ_LEN);

        // 6. DS6: 24 -> 16 (k=5, d=1, p=2) + ECA3 + 残差(h1)
        depthwise_conv1d(buf_A, buf_mid, 24, DSWeights::ds6_depthwise_weight, DSWeights::ds6_depthwise_bias, 5, 1, 2);
        pointwise_conv1d(buf_mid, 24, buf_B, 16, DSWeights::ds6_pointwise_weight, DSWeights::ds6_pointwise_bias, true);
        apply_eca(buf_B, 16, DSWeights::eca3_conv_weight);
        add_residual(buf_B, buf_h1, 16 * SEQ_LEN);

        // 7. Head: Conv1d(16 -> 2, k=3, p=1) 线性输出
        standard_conv1d(buf_B, 16, buf_out, 2, DSWeights::head_weight, DSWeights::head_bias, 3, 1, false);

        std::memcpy(out_i, buf_out, SEQ_LEN * sizeof(float));
        std::memcpy(out_q, buf_out + SEQ_LEN, SEQ_LEN * sizeof(float));
    }

private:
    float buf_in[2 * SEQ_LEN];
    float buf_out[2 * SEQ_LEN];
    float buf_h1[16 * SEQ_LEN];
    float buf_h3[24 * SEQ_LEN];
    float buf_mid[24 * SEQ_LEN];
    float buf_A[24 * SEQ_LEN];
    float buf_B[24 * SEQ_LEN];
};

} // namespace NMR

#endif // M4_EDGE_DS_CNN_HPP

