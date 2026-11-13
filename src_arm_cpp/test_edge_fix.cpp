// =====================================================================
// test_edge_fix.cpp —— 边缘修复上板前验证（PC 端，2026-10-30）
//
// 直接复用 sg21_dual_bench.cpp 的处理函数（#include 进来，宏屏蔽其 main），
// 对真实记录 大豆油24h.iq.bin 做 SG(21,2) 双通道处理，输出 .fix.iq.bin；
// PC 侧再用 numpy 与 scipy 参考逐位 diff。判据：边缘点 rel RMSE 从 ~0.55
// 降到 ≤1e-10，整体 r → 1.000000。
// 编译：g++ -O2 -std=c++11 -o test_edge_fix test_edge_fix.cpp
// =====================================================================
#define SG_BENCH_NO_MAIN
#include "sg21_dual_bench.cpp"
#undef SG_BENCH_NO_MAIN

#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: test_edge_fix in.iq.bin out.iq.bin\n"); return 1; }
    return run_process(argv[1], argv[2]);
}
