// =====================================================================
// m5b_sg_timing_benchmark.cpp —— 板端 SG 滤波计时基准（同板对照表用）
//
// 目的：在 Red Pitaya (Zynq-7020 ARM Cortex-A9) 上实测 Savitzky-Golay
//       (51 点窗, 3 阶) 滤波的单帧耗时与内存占用，与 ECA-DS-TCN
//       Edge 模型（实测 66.8 ms/帧单核 / 35.9 ms/帧双核, VmHWM ~8.4MB）
//       形成同板三维对照表的"SG 行"。
//
// 口径：与 Edge 模型同接口——50kHz 采样流按 1024 点/帧滑动处理。
//       SG 系数在启动时一次性预计算（最小二乘，计时区间之外），
//       滤波本体为 1024×51 MAC 卷积（与部署形态一致）。
//
// 编译（板端，与 Edge 引擎同优化参数）：
//   g++ -O3 -march=armv7-a -mfpu=neon -mfloat-abi=hard
//       -o sg_bench m5b_sg_timing_benchmark.cpp
// 或直接在 PC 端双击 run_sg_bench.bat 一键完成（上传+编译+运行+存日志）。
//
// 无外部依赖（仅 libc/libm），返回 0 表示成功。
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

static const int kWin = 51;     // SG 窗长（与论文冻结基线 SG(51,3) 一致）
static const int kOrder = 3;    // SG 阶数
static const int kFrame = 1024; // 帧长（与 Edge 模型输入一致）
static const int kIters = 2000; // 计时迭代次数

// 启动时预计算 SG 卷积系数（等价于对窗内多项式拟合的平滑值）：
// 解 (V^T V) c = V^T e0，其中 V[j][t] = t^j, e0 = 中心点指示向量。
static bool sg_coefficients(int win, int order, std::vector<double>& coef) {
    const int half = win / 2;
    std::vector<double> V(order + 1), ATA((order + 1) * (order + 1), 0.0),
        ATb(order + 1, 0.0);
    for (int j = 0; j <= order; ++j) V[j] = 0.0; // 占位，下面直接累加
    for (int t = -half; t <= half; ++t) {
        double p = 1.0;
        for (int j = 0; j <= order; ++j) {
            V[j] = p;
            p *= (double)t;
        }
        for (int a = 0; a <= order; ++a)
            for (int b = 0; b <= order; ++b)
                ATA[a * (order + 1) + b] += V[a] * V[b];
        // e0 在 t=0 处取 1（中心点），即平滑系数行
        if (t == 0)
            for (int a = 0; a <= order; ++a) ATb[a] += V[a];
    }
    // 高斯消元求 ATA * coef = ATb
    for (int i = 0; i <= order; ++i) {
        int piv = i;
        for (int r = i + 1; r <= order; ++r)
            if (std::fabs(ATA[r * (order + 1) + i]) >
                std::fabs(ATA[piv * (order + 1) + i]))
                piv = r;
        if (std::fabs(ATA[piv * (order + 1) + i]) < 1e-12) return false;
        if (piv != i)
            for (int c = 0; c <= order; ++c) {
                std::swap(ATA[i * (order + 1) + c], ATA[piv * (order + 1) + c]);
            }
        std::swap(ATb[i], ATb[piv]);
        for (int r = i + 1; r <= order; ++r) {
            double f = ATA[r * (order + 1) + i] / ATA[i * (order + 1) + i];
            for (int c = i; c <= order; ++c)
                ATA[r * (order + 1) + c] -= f * ATA[i * (order + 1) + c];
            ATb[r] -= f * ATb[i];
        }
    }
    coef.assign(win, 0.0);
    // c_j 是 t^j 的多项式系数 → 平滑系数 h[t] = sum_j c_j t^j
    std::vector<double> cj(order + 1);
    for (int i = order; i >= 0; --i) {
        double s = ATb[i];
        for (int j = i + 1; j <= order; ++j)
            s -= ATA[i * (order + 1) + j] * cj[j];
        cj[i] = s / ATA[i * (order + 1) + i];
    }
    for (int t = -half; t <= half; ++t) {
        double p = 1.0, h = 0.0;
        for (int j = 0; j <= order; ++j) {
            h += cj[j] * p;
            p *= (double)t;
        }
        coef[t + half] = h;
    }
    return true;
}

// 计时目标：与部署一致的卷积形态（float, 一维, 1024 点 × 51 抽头）
static void sg_filter_frame(const float* in, float* out,
                            const float* coef, int frame, int win) {
    const int half = win / 2;
    for (int i = 0; i < frame; ++i) {
        double acc = 0.0;
        const int lo = (i < half) ? -i : -half;           // 边界收缩
        const int hi = (i + half >= frame) ? frame - 1 - i : half;
        for (int k = lo; k <= hi; ++k) acc += coef[k + half] * in[i + k];
        out[i] = (float)acc;
    }
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

int main() {
    // 1. 预计算系数（计时区间之外）
    std::vector<double> coef_d;
    if (!sg_coefficients(kWin, kOrder, coef_d)) {
        std::printf("SG coefficient solve failed\n");
        return 1;
    }
    std::vector<float> coef(coef_d.begin(), coef_d.end());

    // 2. 造一段合成包络帧（幅度量级与真实 50kHz 基带一致）
    std::vector<float> in(kFrame), out(kFrame);
    for (int i = 0; i < kFrame; ++i)
        in[i] = std::exp(-(double)i / 300.0) + 0.02f * std::sin(0.37 * i);

    // 3. 预热
    for (int i = 0; i < 50; ++i) sg_filter_frame(in.data(), out.data(),
                                                 coef.data(), kFrame, kWin);
    // 4. 计时（逐次采样，报告 mean/median/min/p90，抗调度抖动）
    std::vector<double> per_call_us(kIters);
    for (int i = 0; i < kIters; ++i) {
        in[0] += 1e-6f; // 防止编译器把重复计算优化掉
        auto s = std::chrono::steady_clock::now();
        sg_filter_frame(in.data(), out.data(), coef.data(), kFrame, kWin);
        auto e = std::chrono::steady_clock::now();
        per_call_us[i] = std::chrono::duration<double, std::micro>(e - s).count();
    }
    std::vector<double> sorted(per_call_us);
    std::sort(sorted.begin(), sorted.end());
    auto pick = [&](double q) {
        return sorted[(int)(q * (kIters - 1))];
    };
    double mean_us = 0.0;
    for (double v : per_call_us) mean_us += v;
    mean_us /= kIters;

    // 5. 汇总（与第4章 Edge 实测口径对齐）
    const int frames_4000echo = 391;
    long hwm = vm_hwm_kib();
    std::printf("=== SG(51,3) board timing benchmark ===\n");
    std::printf("frame points            : %d\n", kFrame);
    std::printf("taps                    : %d (order %d)\n", kWin, kOrder);
    std::printf("iterations              : %d\n", kIters);
    std::printf("SG us per frame (mean)  : %.2f\n", mean_us);
    std::printf("SG us per frame (median): %.2f\n", pick(0.5));
    std::printf("SG us per frame (min)   : %.2f\n", sorted.front());
    std::printf("SG us per frame (p90)   : %.2f\n", pick(0.9));
    std::printf("SG ms, 391-frame rec(med): %.3f (single-core)\n",
                pick(0.5) * 1e-3 * frames_4000echo);
    std::printf("VmHWM                   : %ld kB\n", hwm);
    std::printf("\nReference (Edge model, board-measured, thesis 4.7.1):\n");
    std::printf("  single-core 66.8 ms/frame, dual-core 35.9 ms/frame,\n");
    std::printf("  4000-echo record 26.11 s / 14.04 s, VmHWM 8424/8616 KiB\n");
    // 机器可读行（填表用）：median 口径
    std::printf("RESULT_CSV: sg51_3,%.2f,%.2f,%.2f,%.3f,%ld\n",
                mean_us, pick(0.5), pick(0.9),
                pick(0.5) * 1e-3 * frames_4000echo, hwm);

    // 6. 全长记录模式（与 Edge 基准二进制同负载：200k 点 = 4000 回波×50 点）
    //    使 VmHWM 与 8424/8616 KiB 口径可比（记录数据驻留在进程内）。
    const int kRecord = 200000;
    std::vector<float> rec_in(kRecord), rec_out(kRecord);
    for (int i = 0; i < kRecord; ++i)
        rec_in[i] = (float)(std::exp(-(double)i / 6000.0) +
                            0.05 * std::sin(0.013 * i));
    auto s2 = std::chrono::steady_clock::now();
    sg_filter_frame(rec_in.data(), rec_out.data(), coef.data(), kRecord, kWin);
    auto e2 = std::chrono::steady_clock::now();
    double rec_ms =
        std::chrono::duration<double, std::milli>(e2 - s2).count();
    std::printf("\n=== full-record mode (200k points, same workload class as Edge benchmark) ===\n");
    std::printf("SG full-record time      : %.1f ms (single pass, single-core)\n", rec_ms);
    std::printf("VmHWM after full record  : %ld kB\n", vm_hwm_kib());
    std::printf("RESULT_CSV_FULL: sg51_3,%.1f,%ld\n", rec_ms, vm_hwm_kib());
    return 0;
}
