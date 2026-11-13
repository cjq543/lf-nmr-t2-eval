#!/bin/bash
# 板端全量回放(由主机侧生成, 15条命令显式写死, 避免引号转义问题)
cd /root/replay_20261024
set -e
run() { ./replay_main "data/$1.iq.bin" "data/$1.t.bin" "$2" "out/$1.iq.bin"; }
run 大豆油24h 1.0
run 大豆油24h180度 1.0
run 红花籽油24h180度 1.0
run 大豆油48h 1.0
run 红花籽油48h 1.0
run 大豆油72h180度 1.0
run 红花籽油72h180度 1.0
run 大豆油96h180度 1.0
run 红花籽油96h180度 1.0
run 大豆油未加热 1.0
run 红花籽油未加热 1.0
run 亚麻籽油24h 0.99992
run 红花籽油24h 0.99992
run 菜籽油24h 0.99992
run 葵花籽油24h 0.99992
echo ALL_DONE
