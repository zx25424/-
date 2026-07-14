# step3_causal_identification.py
# -*- coding: utf-8 -*-

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

DATA_DIR = os.getenv("DATA_DIR", "sim_experiment_data")
OUT_DIR = os.getenv("OUT_DIR", os.path.join(DATA_DIR, "new_cot_fairrank"))

STEP3_DIR = os.path.join(OUT_DIR, "step3_causal_outputs")
os.makedirs(STEP3_DIR, exist_ok=True)

COMMON_DIR = os.getenv("COMMON_DIR")
LLM_DIR = os.getenv("LLM_DIR")
OUT_DIR = os.getenv("OUT_DIR")

if COMMON_DIR is None:
    raise ValueError("COMMON_DIR 未设置")

if LLM_DIR is None:
    raise ValueError("LLM_DIR 未设置")

if OUT_DIR is None:
    raise ValueError("OUT_DIR 未设置")

STRUCT_FILE = os.path.join(COMMON_DIR, "T_features_struct.csv")
OUTCOME_FILE = os.path.join(COMMON_DIR, "T_outcomes_v2.csv")
LLMZ_FILE = os.path.join(LLM_DIR, "T_features_llmZ.csv")

OUT_EFFECTS = os.path.join(STEP3_DIR, "task_level_causal_effects.csv")
OUT_SUMMARY = os.path.join(STEP3_DIR, "causal_summary.json")

SEED = int(os.getenv("SEED", "42"))
RHO = float(os.getenv("RHO", "0.5"))

# 奖励干预幅度，可根据论文实验设置修改
SHOCKS = [0.0, 0.1, 0.2]


def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_key_columns(df_list):
    """
    优先使用 task_id + worker_id 合并。
    如果没有 worker_id，则使用 task_id 合并。
    """
    common_cols = set(df_list[0].columns)
    for df in df_list[1:]:
        common_cols &= set(df.columns)

    if "task_id" in common_cols and "worker_id" in common_cols:
        return ["task_id", "worker_id"]
    elif "task_id" in common_cols:
        return ["task_id"]
    else:
        raise KeyError("无法找到合并键。至少需要 task_id，最好同时包含 task_id 和 worker_id。")


def merge_data():
    struct = normalize_columns(safe_read_csv(STRUCT_FILE))
    llmz = normalize_columns(safe_read_csv(LLMZ_FILE))
    outcomes = normalize_columns(safe_read_csv(OUTCOME_FILE))

    key_cols = find_key_columns([struct, llmz, outcomes])

    df = struct.merge(llmz, on=key_cols, how="inner", suffixes=("", "_llm"))
    df = df.merge(outcomes, on=key_cols, how="inner", suffixes=("", "_out"))

    if len(df) == 0:
        raise ValueError("合并后数据为空，请检查 task_id / worker_id 是否一致。")

    print(f"[Merge] 使用合并键: {key_cols}")
    print(f"[Merge] 合并后样本数: {len(df)}")

    return df, key_cols


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def choose_existing_cols(df, candidates):
    return [c for c in candidates if c in df.columns]


def build_features(df):
    """
    X: 任务结构属性
    Z: LLM 语义中介变量
    C: 控制变量
    Y: 结果变量
    """

    x_candidates = [
        "reward", "distance_km", "time_window_sec",
        "route_actual", "route_recommend", "route_diff",
        "task_difficulty", "base_reward", "expected_reward"
    ]

    z_candidates = [
        "semantic_difficulty",
        "environment_complexity",
        "path_uncertainty",
        "rejection_risk",
        "sender_urgency"
    ]

    c_candidates = [
        "peak_flag", "region_density", "weather_risk",
        "worker_load", "historical_accept_rate"
    ]

    y_candidates = ["Uw", "U_w", "worker_utility"]
    s_candidates = ["Us", "Ys", "U_s", "sender_utility"]
    total_candidates = ["Y", "total_utility", "bilateral_utility"]

    X_cols = choose_existing_cols(df, x_candidates)
    Z_cols = choose_existing_cols(df, z_candidates)
    C_cols = choose_existing_cols(df, c_candidates)

    Uw_col = None
    for c in y_candidates:
        if c in df.columns:
            Uw_col = c
            break

    Us_col = None
    for c in s_candidates:
        if c in df.columns:
            Us_col = c
            break

    Y_col = None
    for c in total_candidates:
        if c in df.columns:
            Y_col = c
            break

    if len(X_cols) == 0:
        raise ValueError("未找到任务属性 X 变量，请检查 T_features_struct.csv。")

    if len(Z_cols) == 0:
        raise ValueError("未找到中介变量 Z，请检查 T_features_llmZ.csv。")

    if Uw_col is None:
        raise ValueError("未找到接包人效用列 Uw。")

    if Us_col is None:
        raise ValueError("未找到发包人效用列 Us 或 Ys。")

    if Y_col is None:
        print("[Warn] 未找到综合效用 Y，将根据 Uw 和 Us 重新计算。")
        if "alpha" in df.columns and "eta" in df.columns:
            df["Y"] = df["alpha"] * df[Uw_col] + df["eta"] * df[Us_col]
        else:
            df["Y"] = 0.5 * df[Uw_col] + 0.5 * df[Us_col]
        Y_col = "Y"

    numeric_cols = X_cols + Z_cols + C_cols + [Uw_col, Us_col, Y_col]
    df = ensure_numeric(df, numeric_cols)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    print("[Features] X:", X_cols)
    print("[Features] Z:", Z_cols)
    print("[Features] C:", C_cols)
    print("[Features] Uw:", Uw_col)
    print("[Features] Us:", Us_col)
    print("[Features] Y:", Y_col)
    print("[Features] 有效样本数:", len(df))

    return df, X_cols, Z_cols, C_cols, Uw_col, Us_col, Y_col


def fit_rf(X, y, name):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1
    )

    if len(X) >= 50:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = r2_score(y_test, pred)
    else:
        model.fit(X, y)
        score = None

    print(f"[Model] {name} 训练完成，R2={score}")
    return model, score


def apply_reward_intervention(df, X_cols, shock):
    """
    对 reward 进行干预。
    如果没有 reward 列，则选择第一个 X 变量进行干预。
    """
    X_new = df[X_cols].copy()

    if "reward" in X_new.columns:
        target_col = "reward"
    elif "expected_reward" in X_new.columns:
        target_col = "expected_reward"
    else:
        target_col = X_cols[0]

    X_new[target_col] = X_new[target_col] * (1.0 + shock)

    return X_new, target_col


def estimate_frontdoor(df, X_cols, Z_cols, C_cols, outcome_col, shock):
    """
    简化前门估计：
    1. 学习 E[Z|X]
    2. 学习 E[Y|Z,X,C]
    3. 干预 X 后预测 Z'
    4. 用 Z' 和 X' 预测 Y'
    """

    X_base = df[X_cols]
    Z_base = df[Z_cols]
    y_base = df[outcome_col]

    model_z, r2_z = fit_rf(X_base, Z_base, f"E[Z|X]_{outcome_col}")

    y_features = Z_cols + X_cols + C_cols
    model_y, r2_y = fit_rf(df[y_features], y_base, f"E[{outcome_col}|Z,X,C]")

    X_intervened, intervene_col = apply_reward_intervention(df, X_cols, shock)

    Z_pred_base = model_z.predict(X_base)
    Z_pred_intervened = model_z.predict(X_intervened)

    base_input = pd.DataFrame(Z_pred_base, columns=Z_cols)
    int_input = pd.DataFrame(Z_pred_intervened, columns=Z_cols)

    for c in X_cols:
        base_input[c] = df[c].values
        int_input[c] = X_intervened[c].values

    for c in C_cols:
        base_input[c] = df[c].values
        int_input[c] = df[c].values

    y0 = model_y.predict(base_input[y_features])
    y1 = model_y.predict(int_input[y_features])

    ace = y1 - y0

    return ace, {
        "intervene_col": intervene_col,
        "shock": shock,
        "r2_z": r2_z,
        "r2_y": r2_y
    }


def estimate_gcomp(df, X_cols, Z_cols, C_cols, outcome_col, shock):
    """
    g-computation 反事实估计：
    直接学习 E[Y|X,Z,C]，然后对 X 做干预。
    """
    features = X_cols + Z_cols + C_cols
    y = df[outcome_col]

    model, r2 = fit_rf(df[features], y, f"gcomp_E[{outcome_col}|X,Z,C]")

    X_intervened, intervene_col = apply_reward_intervention(df, X_cols, shock)

    base_df = df[features].copy()
    int_df = df[features].copy()

    for c in X_cols:
        int_df[c] = X_intervened[c].values

    y0 = model.predict(base_df)
    y1 = model.predict(int_df)

    ace = y1 - y0

    return ace, {
        "intervene_col": intervene_col,
        "shock": shock,
        "r2_y": r2
    }


def main():
    df, key_cols = merge_data()
    df, X_cols, Z_cols, C_cols, Uw_col, Us_col, Y_col = build_features(df)

    all_rows = []
    summary = {
        "key_cols": key_cols,
        "X_cols": X_cols,
        "Z_cols": Z_cols,
        "C_cols": C_cols,
        "Uw_col": Uw_col,
        "Us_col": Us_col,
        "Y_col": Y_col,
        "rho": RHO,
        "shocks": SHOCKS,
        "targets": {}
    }

    targets = {
        "Uw": Uw_col,
        "Us": Us_col,
        "Y": Y_col
    }

    result_base = df[key_cols].copy()

    for target_name, target_col in targets.items():
        summary["targets"][target_name] = {}

        for shock in SHOCKS:
            print(f"\n[Estimate] target={target_name}, shock={shock}")

            ace_fd, info_fd = estimate_frontdoor(
                df, X_cols, Z_cols, C_cols, target_col, shock
            )
            ace_gc, info_gc = estimate_gcomp(
                df, X_cols, Z_cols, C_cols, target_col, shock
            )

            ace_star = RHO * ace_fd + (1.0 - RHO) * ace_gc

            result_base[f"ACE_frontdoor_{target_name}_shock_{shock}"] = ace_fd
            result_base[f"ACE_gcomp_{target_name}_shock_{shock}"] = ace_gc
            result_base[f"ACE_star_{target_name}_shock_{shock}"] = ace_star

            summary["targets"][target_name][str(shock)] = {
                "frontdoor_mean": float(np.mean(ace_fd)),
                "frontdoor_std": float(np.std(ace_fd)),
                "gcomp_mean": float(np.mean(ace_gc)),
                "gcomp_std": float(np.std(ace_gc)),
                "ace_star_mean": float(np.mean(ace_star)),
                "ace_star_std": float(np.std(ace_star)),
                "frontdoor_info": info_fd,
                "gcomp_info": info_gc
            }

    # 默认使用 shock=0.1 的 Y 因果效应作为主要排序因果得分
    main_shock = 0.1
    col_main = f"ACE_star_Y_shock_{main_shock}"
    if col_main not in result_base.columns:
        col_main = f"ACE_star_Y_shock_{SHOCKS[-1]}"

    result_base["ACE_star_main"] = result_base[col_main]

    # 把原始关键变量也保留下来，方便 Step4 生成 COT
    keep_cols = key_cols + X_cols + Z_cols + C_cols + [Uw_col, Us_col, Y_col]
    extra_cols = []
    for c in ["alpha", "eta", "rationale", "cot", "initial_cot"]:
        if c in df.columns:
            extra_cols.append(c)

    df_keep = df[keep_cols + extra_cols].copy()
    effects = df_keep.merge(result_base, on=key_cols, how="left")

    effects.to_csv(OUT_EFFECTS, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] Step3 因果识别完成")
    print("[Output]", OUT_EFFECTS)
    print("[Output]", OUT_SUMMARY)


if __name__ == "__main__":
    main()