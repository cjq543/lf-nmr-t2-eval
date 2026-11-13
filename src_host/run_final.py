# -*- coding: utf-8 -*-
"""
run_final.py —— 定稿版(v3)训练驱动: 扩展分布 + 有效窗mask + 每种子独立数据流

协议修订(依据 2026-10-24 OOD 实验暴露的适用范围边界):
  1. 训练分布扩展: 主峰 T2 30~2500ms(原80~900), 峰数1~6(原1~3+长),
     峰宽至0.55(原0.40); 噪声/基线模型不变(原分布优势项保持)。
  2. 30% 随机截断增强(400~2800ms): 输入截断、标签完整、物理损失只算有效窗。
  3. 每种子独立训练数据流(STREAM_SEED+1000*seed): 种子方差覆盖数据随机性。
  4. 传统参数在扩展分布验证集上重新联合搜索后冻结(classical_params_final.json)。
  5. 旧分布行为与旧权重全部不动; 新权重 h1_final_seed{1,2,3}.pth 独立保存。

评价口径:
  - 旧分布主测试(sim_cache tx): 传统分支用旧冻结参数(该分布下的公平基线);
  - 扩展分布测试(final_cache): 传统分支用新冻结参数;
  - 真实15记录: 仅当本地配置有外部原始数据目录时附加(用旧冻结参数, 与v2口径一致)。

用法: python src/run_final.py [--smoke]
"""

import datetime
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from simulator import ExtendedStreamDataset, generate_dataset_ext
from baselines import select_classical_params
from train import build_model, train_sim_trial
import evaluate as EV

SMOKE = "--smoke" in sys.argv
DEVICE = torch.device("cpu")
STATE = os.path.join(H.ROOT_DIR, "run_final_state.json")
LOG = os.path.join(H.LOG_DIR, "run_final.log")
T06_CFG = {"tag": "T06_退火长训_定稿", "stem_dim": 32, "d_model": 128,
           "num_heads": 4, "num_layers": 4, "lambda_huber": 1.2,
           "lambda_wass": 0.7, "lambda_phys": 0.5, "lr": 3e-4}
SEEDS = (1, 2, 3)
if SMOKE:
    STEPS, BS = 300, 16
    N_VAL, N_TEST, N_SEL, N_EVAL = 96, 96, 48, 32
else:
    STEPS, BS = 8000, 32
    N_VAL, N_TEST, N_SEL, N_EVAL = 1024, 1024, 1024, 1024
EXT_VAL_SEED = 303000
EXT_TEST_SEED = 404000


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(H.LOG_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage_done(name):
    if not os.path.exists(STATE):
        return False
    with open(STATE, encoding="utf-8") as f:
        return name in json.load(f).get("completed", [])


def mark_done(name, seconds):
    st = {"completed": []}
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            st = json.load(f)
    st["completed"].append(name)
    st[f"{name}_seconds"] = round(seconds, 1)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def s_cache():
    p = os.path.join(H.RESULT_DIR, "final_cache.npz")
    if os.path.exists(p):
        return
    t0 = time.time()
    vx, vy, vm = generate_dataset_ext(N_VAL, EXT_VAL_SEED)
    tx, ty, tm = generate_dataset_ext(N_TEST, EXT_TEST_SEED)
    np.savez_compressed(p, vx=vx, vy=vy, vm=vm, tx=tx, ty=ty, tm=tm)
    log(f"  扩展分布缓存: 验证{vx.shape} 测试{tx.shape} "
        f"截断占比 {float(np.mean(vm.sum(1) < vm.shape[1])):.0%}, 用时{time.time()-t0:.0f}s")


def s_params():
    p = os.path.join(H.RESULT_DIR, "classical_params_final.json")
    if os.path.exists(p):
        return
    c = np.load(os.path.join(H.RESULT_DIR, "final_cache.npz"))
    from baselines import select_classical_params_masked
    win, poly, lam_sg, lam_dir = select_classical_params_masked(
        c["vx"][:N_SEL], c["vy"][:N_SEL], c["vm"][:N_SEL], log)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"sg_win": int(win), "sg_poly": int(poly),
                   "nnls_lambda": float(lam_sg),
                   "nnls_lambda_direct": float(lam_dir)}, f,
                  ensure_ascii=False, indent=1)


def s_train():
    c = np.load(os.path.join(H.RESULT_DIR, "final_cache.npz"))
    os.makedirs(os.path.join(H.RESULT_DIR, "final"), exist_ok=True)
    for seed in SEEDS:
        ckpt = os.path.join(H.MODEL_DIR, f"h1_final_seed{seed}.pth")
        rec = os.path.join(H.RESULT_DIR, "final", f"seed{seed}.json")
        if os.path.exists(ckpt) and os.path.exists(rec):
            log(f"  seed{seed} 已存在, 跳过")
            continue
        log(f"  -> 定稿训练 seed={seed} (独立数据流 {H.STREAM_SEED + 1000 * seed}, "
            f"{STEPS}步×batch{BS}, CPU)")
        stream = ExtendedStreamDataset(H.STREAM_SEED + 1000 * seed, BS)
        t0 = time.time()
        state, score = train_sim_trial(
            T06_CFG, stream, c["vx"], c["vy"], DEVICE, STEPS,
            1500, 5, log, seed=seed, val_masks=c["vm"])
        torch.save({"model_state_dict": state, "config": T06_CFG,
                    "seed": seed, "ext_val_score": score, "protocol": "v3扩展分布"},
                   ckpt)
        with open(rec, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "ext_val_score": score,
                       "steps": STEPS, "stream_seed": H.STREAM_SEED + 1000 * seed},
                      f, ensure_ascii=False, indent=1)
        log(f"  seed{seed} 完成: 扩展验证分 {score:.2f}, 用时{time.time()-t0:.0f}s")


def s_eval():
    import pandas as pd
    fin = np.load(os.path.join(H.RESULT_DIR, "final_cache.npz"))
    old = np.load(os.path.join(H.RESULT_DIR, "sim_cache.npz"))
    with open(os.path.join(H.RESULT_DIR, "classical_params_masked_20261025.json"),
              encoding="utf-8") as f:
        cpf = json.load(f)
    with open(os.path.join(H.RESULT_DIR, "classical_params.json"),
              encoding="utf-8") as f:
        cpo = json.load(f)

    models = {}
    ck_old = torch.load(os.path.join(H.MODEL_DIR, "h1_T06_退火长训.pth"),
                        map_location="cpu", weights_only=True)
    m = build_model(ck_old["config"]); m.load_state_dict(ck_old["model_state_dict"])
    models["A旧版(原分布)"] = m
    scores = {}
    for seed in SEEDS:
        p = os.path.join(H.MODEL_DIR, f"h1_final_seed{seed}.pth")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True)
        m = build_model(ck["config"]); m.load_state_dict(ck["model_state_dict"])
        models[f"A定稿seed{seed}"] = m
        scores[seed] = ck["ext_val_score"]

    sim_tables = {}
    n = N_EVAL if not SMOKE else N_EVAL
    # 旧分布主测试: 传统分支用旧冻结参数
    br_old = EV.make_branches(cpo["sg_win"], cpo["sg_poly"], cpo["nnls_lambda"],
                              models, DEVICE,
                              nnls_lambda_direct=cpo.get(
                                  "nnls_lambda_direct", cpo["nnls_lambda"]))
    rows_old, _ = EV.run_branches(list(old["tx"][:n]), list(old["ty"][:n]),
                                  br_old, DEVICE)
    sim_tables[f"旧分布主测试集(n={n}, 传统分支用旧冻结参数)"] = rows_old
    # 扩展分布测试: 传统分支用新冻结参数
    br_fin = EV.make_branches(cpf["sg_win"], cpf["sg_poly"], cpf["nnls_lambda"],
                              models, DEVICE,
                              nnls_lambda_direct=cpf.get(
                                  "nnls_lambda_direct", cpf["nnls_lambda"]))
    rows_ext, preds_ext = EV.run_branches(list(fin["tx"][:n]), list(fin["ty"][:n]),
                                          br_fin, DEVICE, masks=fin["tm"][:n])
    sim_tables[f"扩展分布测试集(n={n}, 含30%截断, 传统分支用新冻结参数)"] = rows_ext

    real_rows = []
    if H.REAL_DATA_AVAILABLE:
        from realdata import load_real_records
        records = load_real_records()
        echoes = [r["echo"] for r in records]
        refs = [r["spec"] for r in records]
        rr, _ = EV.run_branches(
            echoes, refs, br_old, DEVICE,
            masks=[r_['valid_mask'] for r_ in records])
        nm = {i: f"{r['name']}|{r['split']}" for i, r in enumerate(records)}
        for row in rr:
            row["sample_name"], row["split"] = nm[row["sample"]].split("|")
        real_rows = rr

    pd.DataFrame(rows_old).to_csv(
        os.path.join(H.RESULT_DIR, "final_old_dist_rows.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_ext).to_csv(
        os.path.join(H.RESULT_DIR, "final_ext_dist_rows.csv"),
        index=False, encoding="utf-8-sig")
    if real_rows:
        pd.DataFrame(real_rows).to_csv(
            os.path.join(H.RESULT_DIR, "final_real15_rows.csv"),
            index=False, encoding="utf-8-sig")

    for name, rows in sim_tables.items():
        g = pd.DataFrame(rows).groupby("branch")[EV.METRIC_KEYS].mean().round(4)
        log(f"\n[{name}]\n{g.to_string()}")

    best_seed = max(scores, key=scores.get) if scores else None
    summary = [
        f"- 生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M} (v3 定稿版)",
        f"- 协议: 扩展分布(主峰30~2500ms/1~6峰/峰宽至0.55) + 30%随机截断增强"
        f"(物理损失仅有效窗) + 每种子独立数据流; 噪声模型与旧版一致",
        f"- 定稿种子扩展验证分: {scores}",
        f"- 传统参数: 扩展分布冻结(直通λ={cpf.get('nnls_lambda_direct')}, "
        f"SG({cpf['sg_win']},{cpf['sg_poly']})+λ={cpf['nnls_lambda']}); "
        f"旧分布表沿用旧冻结参数",
        f"- 旧版A与旧结果未被覆盖; 新权重 h1_final_seed*.pth",
    ]
    if best_seed:
        summary.append(f"- OOD复测建议命令: python src/ood_evaluate.py "
                       f"--model-a h1_final_seed{best_seed}.pth --model-b none "
                       f"--out-suffix _final")
    EV.write_report(sim_tables, real_rows, summary,
                    os.path.join(H.ROOT_DIR, "最终报告_v3定稿版.md"))
    log(f"  报告已写入 最终报告_v3定稿版.md")


def main():
    log("=" * 64)
    log(f"定稿版(v3)训练启动 SMOKE={SMOKE} | 设备: CPU-only")
    log("=" * 64)
    for name, fn in (("f_cache", s_cache), ("f_params", s_params),
                     ("f_train", s_train), ("f_eval", s_eval)):
        if stage_done(name):
            log(f"== 阶段 {name} 已完成, 跳过 ==")
            continue
        log(f"==== 阶段 {name} 开始 ====")
        t0 = time.time()
        fn()
        mark_done(name, time.time() - t0)
        log(f"==== 阶段 {name} 完成, 用时 {time.time() - t0:.0f}s ====")
    log("定稿版全部阶段结束。")


if __name__ == "__main__":
    main()
