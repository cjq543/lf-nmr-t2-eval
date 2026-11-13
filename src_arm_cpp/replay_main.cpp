// replay_main.cpp —— 板端真实Fid回放主程序(2026-10-24)
// 协议逐位复刻 PC 侧 cascade_experiment.apply_edge_model_streaming:
//   反射填充 win/2 + 滑窗1024/步长512 + 逐窗双通道max归一(下限1e-6)
//   + 候选掩码(死区[0.240,0.300]ms mod TE) + 残差 clean=x-Net(x)*mask
//   + 汉宁窗 OLA; 输出剥离填充区。
// 用法: ./replay_main in.iq.bin in.t.bin te_ms out.iq.bin
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "m4_edge_ds_cnn.hpp"

static constexpr int WIN = 1024;
static constexpr int HOP = 512;

static std::vector<double> np_hanning(int n) {   // numpy.hanning(对称, float64)
    std::vector<double> w(n);
    for (int k = 0; k < n; ++k)
        w[k] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / (n - 1));
    return w;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s in.iq.bin in.t.bin te_ms out.iq.bin\n", argv[0]); return 1; }
    const char* in_path = argv[1];
    const char* t_path = argv[2];
    const double te_ms = std::atof(argv[3]);
    const char* out_path = argv[4];

    // ---- 读输入 ----
    FILE* f = std::fopen(in_path, "rb");
    if (!f) { std::perror("open in"); return 1; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    int n_pts = (int)(sz / (2 * sizeof(float)));   // 交错 I,Q float32
    std::vector<float> raw(sz / sizeof(float));
    if (std::fread(raw.data(), sizeof(float), raw.size(), f) != raw.size()) { std::fprintf(stderr, "read fail\n"); return 1; }
    std::fclose(f);

    std::vector<double> times(n_pts);
    f = std::fopen(t_path, "rb");
    if (!f || std::fread(times.data(), sizeof(double), n_pts, f) != (size_t)n_pts) { std::fprintf(stderr, "read t fail\n"); return 1; }
    std::fclose(f);

    // ---- 拆通道 + 反射填充 win/2 ----
    const int PAD = WIN / 2;
    const int total = n_pts + 2 * PAD;
    std::vector<float> pi_(total), pq(total);
    std::vector<double> pt(total);
    for (int i = 0; i < n_pts; ++i) {
        pi_[PAD + i] = raw[2 * i];
        pq[PAD + i] = raw[2 * i + 1];
        pt[PAD + i] = times[i];
    }
    for (int i = 0; i < PAD; ++i) {              // 左侧 numpy reflect: x[pad-i]
        pi_[i] = pi_[2 * PAD - i];
        pq[i] = pq[2 * PAD - i];
        pt[i] = pt[PAD];                          // 时间轴边缘填充(np.pad 'edge')
    }
    for (int i = 0; i < PAD; ++i) {              // 右侧 reflect: x[n-2-i]
        pi_[total - 1 - i] = pi_[PAD + n_pts - 2 - i];
        pq[total - 1 - i] = pq[PAD + n_pts - 2 - i];
        pt[total - 1 - i] = pt[PAD + n_pts - 1];
    }

    const std::vector<double> hann = np_hanning(WIN);
    std::vector<double> acc_i(total, 0.0), acc_q(total, 0.0), acc_w(total, 0.0);

    std::vector<int> starts;
    for (int st = 0; st + WIN <= total; st += HOP) starts.push_back(st);

    auto t0 = std::chrono::steady_clock::now();
    int n_frames = 0;
# pragma omp parallel for schedule(dynamic) reduction(+:n_frames)
    for (int si = 0; si < (int)starts.size(); ++si) {
        const int st = starts[si];
        NMR::EdgeDSTCN engine;                    // 成员缓冲非线程安全: 每帧独立实例
        float ci[WIN], cq[WIN];
        std::memcpy(ci, &pi_[st], WIN * sizeof(float));
        std::memcpy(cq, &pq[st], WIN * sizeof(float));

        float sc = 1e-6f;
        for (int k = 0; k < WIN; ++k) {
            float a = std::fabs(ci[k]); if (a > sc) sc = a;
            float b = std::fabs(cq[k]); if (b > sc) sc = b;
        }
        for (int k = 0; k < WIN; ++k) { ci[k] /= sc; cq[k] /= sc; }

        float ni[WIN], nq[WIN];
        engine.forward(ci, cq, ni, nq);

        // 候选掩码: 死区 [0.240,0.300] ms (mod TE) 内掩码=0
        float clean_i[WIN], clean_q[WIN];
        for (int k = 0; k < WIN; ++k) {
            double phase = std::fmod(pt[st + k], te_ms);
            bool dead = (phase >= 0.240 - 1e-9) && (phase <= 0.300 + 1e-9);
            float m = dead ? 0.0f : 1.0f;
            clean_i[k] = ci[k] - ni[k] * m;
            clean_q[k] = cq[k] - nq[k] * m;
        }
# pragma omp critical
        for (int k = 0; k < WIN; ++k) {
            double h = hann[k];
            acc_i[st + k] += (double)clean_i[k] * (double)sc * h;
            acc_q[st + k] += (double)clean_q[k] * (double)sc * h;
            acc_w[st + k] += h;
        }
        ++n_frames;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // ---- 归一化 + 剥离填充, 写出交错 float32 ----
    std::vector<float> out(2 * n_pts);
    for (int i = 0; i < n_pts; ++i) {
        int j = PAD + i;
        double w = acc_w[j] > 1e-6 ? acc_w[j] : 1.0;
        out[2 * i] = (float)(acc_i[j] / w);
        out[2 * i + 1] = (float)(acc_q[j] / w);
    }
    f = std::fopen(out_path, "wb");
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);

    std::printf("replay_ok n_pts=%d frames=%d te_ms=%.5f secs=%.3f "
                "ms_per_frame=%.2f threads=%d\n",
                n_pts, n_frames, te_ms, secs,
                secs * 1000.0 / (n_frames ? n_frames : 1), omp_get_max_threads());
    return 0;
}
