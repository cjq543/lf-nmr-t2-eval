// =====================================================================
// sg21_dual_bench.cpp —— 同配置 SG(21,2) 双通道板端基准（改稿清单第 5 步）
//
// 目的：补齐"质量-成本同配置"缺口。质量实验（cascade_lowsnr.py）的 SG 为
//   scipy savgol_filter(x[ch], 21, 2) 逐通道（默认 mode='interp' 边界）；
//   本程序在 Red Pitaya (Zynq-7020 ARM Cortex-A9) 上以同一协议实测
//   双通道 SG(21,2) 的耗时/内存，并对真实记录输出滤波结果供 PC diff。
//
// 两种模式：
//   1) 计时基准（无参数）：合成信号，逐帧 + 全长双口径，两次运行取中位
//      （沿用 sg_bench 20261026 两次独立运行 <0.1% 的纪律）；
//   2) 处理真实记录：--in <x.iq.bin> --out <y.iq.bin>
//      输入 = float32 交错 I/Q（50kHz 对齐，与 01_准备上传数据.py 逐字节
//      同格式），逐通道 SG(21,2)，输出同格式 float32。
//
// 边界口径（关键）：与 scipy mode='interp' 逐位对齐——
//   内部点：SG 中心行系数（最小二乘预计算，与旧 bench 同解法）；
//   两端各 half=10 点：对最前/最后 21 点做 2 阶多项式最小二乘拟合，
//   在边缘位置求值（scipy _fit_edge 语义），双精度求解。
//
// 编译（板端，与 Edge 引擎/旧 sg_bench 同参数）：
//   g++ -std=c++11 -O3 -march=armv7-a -mfpu=neon -mfloat-abi=hard \
//       -o sg21_bench sg21_dual_bench.cpp
// 无外部依赖（仅 libc/libm）。返回 0 = 成功。
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

static const int kWin = 21;      // SG 窗长（与质量实验冻结配置一致）
static const int kOrder = 2;     // SG 阶数
static const int kFrame = 1024;  // 帧长（与 Edge/旧 bench 口径一致）
static const int kIters = 2000;  // 计时迭代次数

// ---------- SG 中心行系数（最小二乘，双精度，同旧 bench 解法） ----------
static bool sg_center_coefficients(int win, int order, std::vector<double>& coef) {
    const int half = win / 2;
    std::vector<double> V(order + 1), ATA((order + 1) * (order + 1), 0.0),
        ATb(order + 1, 0.0);
    for (int t = -half; t <= half; ++t) {
        double p = 1.0;
        for (int j = 0; j <= order; ++j) { V[j] = p; p *= (double)t; }
        for (int a = 0; a <= order; ++a)
            for (int b = 0; b <= order; ++b)
                ATA[a * (order + 1) + b] += V[a] * V[b];
        if (t == 0)
            for (int a = 0; a <= order; ++a) ATb[a] += V[a];
    }
    for (int i = 0; i <= order; ++i) {
        int piv = i;
        for (int r = i + 1; r <= order; ++r)
            if (std::fabs(ATA[r * (order + 1) + i]) >
                std::fabs(ATA[piv * (order + 1) + i])) piv = r;
        if (std::fabs(ATA[piv * (order + 1) + i]) < 1e-12) return false;
        if (piv != i)
            for (int c = 0; c <= order; ++c)
                std::swap(ATA[i * (order + 1) + c], ATA[piv * (order + 1) + c]);
        std::swap(ATb[i], ATb[piv]);
        for (int r = i + 1; r <= order; ++r) {
            double f = ATA[r * (order + 1) + i] / ATA[i * (order + 1) + i];
            for (int c = i; c <= order; ++c) ATA[r * (order + 1) + c] -= f * ATA[i * (order + 1) + c];
            ATb[r] -= f * ATb[i];
        }
    }
    std::vector<double> cj(order + 1);
    for (int i = order; i >= 0; --i) {
        double s = ATb[i];
        for (int j = i + 1; j <= order; ++j) s -= ATA[i * (order + 1) + j] * cj[j];
        cj[i] = s / ATA[i * (order + 1) + i];
    }
    coef.assign(win, 0.0);
    for (int t = -half; t <= half; ++t) {
        double p = 1.0, h = 0.0;
        for (int j = 0; j <= order; ++j) { h += cj[j] * p; p *= (double)t; }
        coef[t + half] = h;
    }
    return true;
}

// ---------- scipy mode='interp' 的边缘求值（对 21 点窗拟合 2 阶多项式） ----------
// window = x[base .. base+win-1]，在窗口内位置 pos 处求拟合值（t=0..win-1）。
// 2026-10-30 修复：原版对 3×3 对称满秩正规方程直接回代、漏做前向消元，
// 边缘点解出错误二次式（与 scipy 相对误差 ~100%，整体 r 被拖到 0.9981）。
// 现补全带列主元的完整消元 + 回代；与 scipy/numpy lstsq 一致到 1e-12
// （验证脚本 _edge_convention_probe.py，C1/R1 口径）。
static double sg_interp_edge_point(const float* window, int pos) {
    // 最小二乘：V[t,j] = t^j (t=0..win-1, j=0..order)，解 V^T V c = V^T y
    double ATA[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // (order+1)^2 = 9
    double ATb[3] = {0, 0, 0};
    for (int t = 0; t < kWin; ++t) {
        double p = 1.0;
        double v[3];
        for (int j = 0; j <= kOrder; ++j) { v[j] = p; p *= (double)t; }
        for (int a = 0; a <= kOrder; ++a) {
            for (int b = 0; b <= kOrder; ++b) ATA[a * 3 + b] += v[a] * v[b];
            ATb[a] += v[a] * (double)window[t];
        }
    }
    // 高斯消元（列主元）+ 回代——完整求解，不得省略消元步骤
    for (int i = 0; i <= kOrder; ++i) {
        int piv = i;
        for (int r = i + 1; r <= kOrder; ++r)
            if (std::fabs(ATA[r * 3 + i]) > std::fabs(ATA[piv * 3 + i]))
                piv = r;
        if (std::fabs(ATA[piv * 3 + i]) < 1e-12) return 0.0;
        if (piv != i) {
            for (int c = 0; c <= kOrder; ++c)
                std::swap(ATA[i * 3 + c], ATA[piv * 3 + c]);
            std::swap(ATb[i], ATb[piv]);
        }
        for (int r = i + 1; r <= kOrder; ++r) {
            double f = ATA[r * 3 + i] / ATA[i * 3 + i];
            for (int c = i; c <= kOrder; ++c) ATA[r * 3 + c] -= f * ATA[i * 3 + c];
            ATb[r] -= f * ATb[i];
        }
    }
    double cj[3] = {0, 0, 0};
    for (int i = kOrder; i >= 0; --i) {
        double s = ATb[i];
        for (int j = i + 1; j <= kOrder; ++j) s -= ATA[i * 3 + j] * cj[j];
        cj[i] = s / ATA[i * 3 + i];
    }
    double out = 0.0, p = 1.0;
    for (int j = 0; j <= kOrder; ++j) { out += cj[j] * p; p *= (double)pos; }
    return out;
}

// ---------- 逐通道 SG(21,2)：内部卷积 + interp 边缘 ----------
static void sg_filter_channel(const float* in, float* out, int n,
                              const float* coef) {
    const int half = kWin / 2;
    for (int i = half; i < n - half; ++i) {
        double acc = 0.0;
        for (int k = -half; k <= half; ++k) acc += coef[k + half] * in[i + k];
        out[i] = (float)acc;
    }
    // 左边缘 [0, half)：对 x[0..kWin-1] 拟合，在位置 i 求值
    for (int i = 0; i < half && i < n; ++i)
        out[i] = (float)sg_interp_edge_point(in, i);
    // 右边缘 [n-half, n)：对 x[n-kWin..n-1] 拟合，在窗口内位置 i-(n-kWin) 求值
    for (int i = (n - half > half) ? n - half : half; i < n; ++i)
        out[i] = (float)sg_interp_edge_point(in + (n - kWin), i - (n - kWin));
}

// ---------- 交错 I/Q 整体处理 ----------
static void sg_filter_iq(const float* in, float* out, long n_pts,
                         const float* coef) {
    // I = 偶数位，Q = 奇数位
    std::vector<float> chI(n_pts), chQ(n_pts), oI(n_pts), oQ(n_pts);
    for (long i = 0; i < n_pts; ++i) { chI[i] = in[2 * i]; chQ[i] = in[2 * i + 1]; }
    sg_filter_channel(chI.data(), oI.data(), (int)n_pts, coef);
    sg_filter_channel(chQ.data(), oQ.data(), (int)n_pts, coef);
    for (long i = 0; i < n_pts; ++i) { out[2 * i] = oI[i]; out[2 * i + 1] = oQ[i]; }
}

static long vm_hwm_kib() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long hwm = -1;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "VmHWM: %ld kB", &hwm) == 1) break;
    std::fclose(f);
    return hwm;
}

static int run_bench() {
    std::vector<double> coef_d;
    if (!sg_center_coefficients(kWin, kOrder, coef_d)) {
        std::printf("SG coefficient solve failed\n"); return 1;
    }
    std::vector<float> coef(coef_d.begin(), coef_d.end());

    // 合成双通道帧（量级与真实 50kHz 基带一致）
    std::vector<float> in(2 * kFrame), out(2 * kFrame);
    for (int i = 0; i < kFrame; ++i) {
        in[2 * i]     = std::exp(-(double)i / 300.0) + 0.02f * std::sin(0.37 * i);
        in[2 * i + 1] = 0.03f * std::cos(0.29 * i);
    }
    for (int i = 0; i < 50; ++i)
        sg_filter_iq(in.data(), out.data(), kFrame, coef.data());

    std::vector<double> per_call_us(kIters);
    for (int i = 0; i < kIters; ++i) {
        in[0] += 1e-6f;
        auto s = std::chrono::steady_clock::now();
        sg_filter_iq(in.data(), out.data(), kFrame, coef.data());
        auto e = std::chrono::steady_clock::now();
        per_call_us[i] = std::chrono::duration<double, std::micro>(e - s).count();
    }
    std::vector<double> sorted(per_call_us);
    std::sort(sorted.begin(), sorted.end());
    auto pick = [&](double q) { return sorted[(int)(q * (kIters - 1))]; };
    double mean_us = 0.0;
    for (double v : per_call_us) mean_us += v;
    mean_us /= kIters;

    const int frames_4000echo = 391;
    long hwm = vm_hwm_kib();
    std::printf("=== SG(21,2) dual-channel board timing benchmark ===\n");
    std::printf("frame points            : %d x 2 ch (interleaved I/Q)\n", kFrame);
    std::printf("taps                    : %d (order %d)\n", kWin, kOrder);
    std::printf("iterations              : %d\n", kIters);
    std::printf("SG us per frame (mean)  : %.2f\n", mean_us);
    std::printf("SG us per frame (median): %.2f\n", pick(0.5));
    std::printf("SG us per frame (min)   : %.2f\n", sorted.front());
    std::printf("SG us per frame (p90)   : %.2f\n", pick(0.9));
    std::printf("SG ms, 391-frame rec(med): %.3f (single-core, dual-ch total)\n",
                pick(0.5) * 1e-3 * frames_4000echo);
    std::printf("VmHWM                   : %ld kB\n", hwm);
    std::printf("RESULT_CSV: sg21_2_dual,%.2f,%.2f,%.2f,%.3f,%ld\n",
                mean_us, pick(0.5), pick(0.9),
                pick(0.5) * 1e-3 * frames_4000echo, hwm);

    // 全长模式：2 x 200,000 点（与旧 bench 同负载级，双通道）
    const long kRecord = 200000;
    std::vector<float> rec_in(2 * kRecord), rec_out(2 * kRecord);
    for (long i = 0; i < kRecord; ++i) {
        rec_in[2 * i]     = (float)(std::exp(-(double)i / 6000.0) + 0.05 * std::sin(0.013 * i));
        rec_in[2 * i + 1] = (float)(0.04 * std::cos(0.011 * i));
    }
    auto s2 = std::chrono::steady_clock::now();
    sg_filter_iq(rec_in.data(), rec_out.data(), kRecord, coef.data());
    auto e2 = std::chrono::steady_clock::now();
    double rec_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();
    std::printf("\n=== full-record mode (2 x 200k pts, dual-channel) ===\n");
    std::printf("SG full-record time      : %.1f ms (single pass, single-core)\n", rec_ms);
    std::printf("VmHWM after full record  : %ld kB\n", vm_hwm_kib());
    std::printf("RESULT_CSV_FULL: sg21_2_dual,%.1f,%ld\n", rec_ms, vm_hwm_kib());
    return 0;
}

static int run_process(const char* in_path, const char* out_path) {
    std::FILE* f = std::fopen(in_path, "rb");
    if (!f) { std::printf("cannot open %s\n", in_path); return 1; }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    long n_pts = bytes / (long)(2 * sizeof(float));  // 交错 I/Q
    std::vector<float> in(2 * n_pts), out(2 * n_pts);
    if (std::fread(in.data(), sizeof(float), 2 * n_pts, f) != (size_t)(2 * n_pts)) {
        std::printf("short read on %s\n", in_path); std::fclose(f); return 1;
    }
    std::fclose(f);

    std::vector<double> coef_d;
    if (!sg_center_coefficients(kWin, kOrder, coef_d)) return 1;
    std::vector<float> coef(coef_d.begin(), coef_d.end());

    auto s = std::chrono::steady_clock::now();
    sg_filter_iq(in.data(), out.data(), n_pts, coef.data());
    auto e = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(e - s).count();

    std::FILE* g = std::fopen(out_path, "wb");
    if (!g) { std::printf("cannot write %s\n", out_path); return 1; }
    std::fwrite(out.data(), sizeof(float), 2 * n_pts, g);
    std::fclose(g);

    double sum = 0.0;
    for (long i = 0; i < 2 * n_pts; ++i) sum += (double)out[i];
    std::printf("PROC: %s n_pts=%ld ms=%.3f VmHWM=%ld kB checksum=%.6f\n",
                out_path, n_pts, ms, vm_hwm_kib(), sum);
    return 0;
}

#ifndef SG_BENCH_NO_MAIN
int main(int argc, char** argv) {
    const char* in_path = nullptr;
    const char* out_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
            in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!in_path) {
            in_path = argv[i];
        } else if (!out_path) {
            out_path = argv[i];
        }
    }
    if (in_path && out_path) {
        return run_process(in_path, out_path);
    }
    return run_bench();
}
#endif // SG_BENCH_NO_MAIN
