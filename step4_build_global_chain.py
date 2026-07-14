# -*- coding: utf-8 -*-
"""
Step4：构建“全局优化因果排序思维链”与“候选任务级因果证据表”

用途：
1. 保留 Step2、Step3 的已有结果，不重新生成语义变量和因果效应；
2. 将结构化特征、LLM 语义变量、双边效用、任务级因果效应合并为 task_ranking_evidence.csv；
3. 生成一条 initial causal ranking chain；
4. 生成一条 global optimized causal ranking chain。

运行示例：
python step4_build_global_chain.py --base_dir E:\sim_experiment_data --model deepseek
python step4_build_global_chain.py --base_dir E:\sim_experiment_data --model doubao
python step4_build_global_chain.py --base_dir E:\sim_experiment_data --model gpt4o
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """兼容 utf-8-sig / utf-8 / gbk 编码读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_task_id(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """统一任务ID字段为 task_id。"""
    candidates = ["task_id", "taskid", "taskId", "id", "ID", "任务ID", "任务编号"]
    for c in candidates:
        if c in df.columns:
            if c != "task_id":
                df = df.rename(columns={c: "task_id"})
            return df
    raise ValueError(f"{table_name} 中找不到任务ID字段。已有字段：{list(df.columns)}")


def rename_by_candidates(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """根据候选字段名统一列名。若目标列已存在，则不覆盖。"""
    rename = {}
    cols = set(df.columns)
    for std_col, candidates in mapping.items():
        if std_col in cols:
            continue
        for c in candidates:
            if c in cols:
                rename[c] = std_col
                cols.add(std_col)
                break
    return df.rename(columns=rename)


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    merge 后可能出现 col_x / col_y。这里将其合并为 col。
    优先保留已有 col，然后依次用 col_x、col_y 补空值。
    """
    suffixes = ["_x", "_y", "_struct", "_llmz", "_outcome", "_causal"]
    base_names = set()
    for col in df.columns:
        for suf in suffixes:
            if col.endswith(suf):
                base_names.add(col[: -len(suf)])
    for base in base_names:
        related = [base] if base in df.columns else []
        related += [f"{base}{suf}" for suf in suffixes if f"{base}{suf}" in df.columns]
        if not related:
            continue
        series = df[related[0]]
        for c in related[1:]:
            series = series.combine_first(df[c])
        df[base] = series
        drop_cols = [c for c in related if c != base]
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def to_numeric_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_01(s: pd.Series) -> pd.Series:
    """
    将风险类字段转成 0~1。
    - 若已经在 0~1，则保持；
    - 若大概率为 1~5 打分，则除以 5；
    - 其他情况用 min-max。
    """
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mn >= 0 and mx <= 1:
        return x
    if mn >= 1 and mx <= 5:
        return x / 5.0
    if mx == mn:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def first_existing_file(candidates: List[Path], required: bool = True) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    if required:
        raise FileNotFoundError("以下候选文件均不存在：\n" + "\n".join(map(str, candidates)))
    return None


def round_numeric(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def build_step4_outputs(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    model = args.model

    common_dir = base_dir / "common"
    llm_dir = base_dir / model / args.step2_dir_name
    step3_dir = base_dir / model / args.step3_dir_name
    out_dir = base_dir / model / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    struct_path = first_existing_file([
        common_dir / "T_features_struct.csv",
        common_dir / "features_struct.csv",
        common_dir / "T_struct.csv",
    ])
    llmz_path = first_existing_file([
        llm_dir / "T_features_llmZ.csv",
        llm_dir / "features_llmZ.csv",
        llm_dir / "T_llmZ.csv",
    ])
    outcomes_path = first_existing_file([
        common_dir / "T_outcomes_v2.csv",
        common_dir / "T_outcomes.csv",
        common_dir / "outcomes.csv",
    ])
    causal_path = first_existing_file([
        step3_dir / "task_level_causal_effects.csv",
        step3_dir / "T_task_level_causal_effects.csv",
        step3_dir / "causal_effects.csv",
    ])

    struct = ensure_task_id(read_csv_flexible(struct_path), "结构化特征表")
    llmz = ensure_task_id(read_csv_flexible(llmz_path), "语义中介变量表")
    outcomes = ensure_task_id(read_csv_flexible(outcomes_path), "双边效用结果表")
    causal = ensure_task_id(read_csv_flexible(causal_path), "因果效应表")

    common_mapping = {
        "reward": ["base_fee", "fee", "task_reward", "reward_fee", "报酬", "任务报酬"],
        "distance_km": ["distance", "delivery_distance", "route_distance", "配送距离", "距离"],
        "time_window_sec": ["time_window", "time_window_seconds", "window_sec", "时间窗", "时间窗长度"],
        "task_difficulty_struct": ["task_difficulty", "difficulty", "struct_difficulty", "任务难度"],
        "path_error": ["route_diff_percent", "path_diff", "route_error", "路径差异"],
        "region_density": ["density", "区域密度"],
        "peak_flag": ["is_peak", "peak", "高峰期"],
    }

    llmz_mapping = {
        "semantic_difficulty": ["semantic_diff", "sem_difficulty", "语义难度"],
        "environment_complexity": ["env_complexity", "environment", "环境复杂度"],
        "path_uncertainty": ["route_uncertainty", "path_risk", "路径不确定性"],
        "rejection_risk": ["reject_risk", "拒单风险"],
        "sender_urgency": ["urgency", "发包紧急度"],
        "alpha": ["worker_weight", "alpha_w"],
        "eta": ["sender_weight", "eta_s"],
    }

    outcome_mapping = {
        "Uw": ["U_w", "worker_utility", "utility_worker", "接包人效用"],
        "Us": ["Ys", "U_s", "sender_utility", "utility_sender", "发包人效用"],
        "Y": ["bilateral_utility", "utility_total", "综合效用", "Y_bilateral"],
        "delay_proxy": ["delay", "等待延迟", "延迟"],
    }

    causal_mapping = {
        # 前门调整因果效应，优先使用双边综合效用 Y 在 shock=0.1 下的结果
        "ACE_fd": [
            "ACEfd",
            "ace_fd",
            "frontdoor_ace",
            "ACE_frontdoor",
            "ACE_frontdoor_Y_shock_0.1",
            "ACE_frontdoor_Y_shock_0.0",
            "ACE_frontdoor_Y_shock_0.2",
        ],

        # g-computation 因果效应，优先使用双边综合效用 Y 在 shock=0.1 下的结果
        "ACE_gc": [
            "ACEgc",
            "ace_gc",
            "gcomp_ace",
            "ACE_gcomp",
            "ACE_gcomp_Y_shock_0.1",
            "ACE_gcomp_Y_shock_0.0",
            "ACE_gcomp_Y_shock_0.2",
        ],

        # 融合因果效应，优先使用旧 Step3 已经生成的 ACE_star_main
        # 如果没有 ACE_star_main，则使用 Y 的融合因果效应
        "ACEstar": [
            "ACEstar",
            "ACE_star_main",
            "ACE_star",
            "ACE*",
            "ACE_fused",
            "ace_star",
            "ace_fused",
            "ACE",
            "ACE_star_Y_shock_0.1",
            "ACE_star_Y_shock_0.0",
            "ACE_star_Y_shock_0.2",
        ],

        # 接包人侧融合因果效应
        "ACEw_star": [
            "ACE_worker_star",
            "ACEw",
            "ACE_Uw",
            "ACE_star_Uw_shock_0.1",
            "ACE_star_Uw_shock_0.0",
            "ACE_star_Uw_shock_0.2",
        ],

        # 发包人侧融合因果效应
        "ACEs_star": [
            "ACE_sender_star",
            "ACEs",
            "ACE_Us",
            "ACE_Ys",
            "ACE_star_Us_shock_0.1",
            "ACE_star_Us_shock_0.0",
            "ACE_star_Us_shock_0.2",
        ],
    }

    struct = rename_by_candidates(struct, common_mapping)
    llmz = rename_by_candidates(llmz, llmz_mapping)
    outcomes = rename_by_candidates(outcomes, outcome_mapping)
    causal = rename_by_candidates(causal, causal_mapping)

    df = struct.merge(llmz, on="task_id", how="inner", suffixes=("_struct", "_llmz"))
    df = df.merge(outcomes, on="task_id", how="inner", suffixes=("", "_outcome"))
    df = df.merge(causal, on="task_id", how="inner", suffixes=("", "_causal"))
    df = coalesce_duplicate_columns(df)

    numeric_candidates = [
        "reward", "distance_km", "time_window_sec", "task_difficulty_struct", "path_error",
        "semantic_difficulty", "environment_complexity", "path_uncertainty", "rejection_risk", "sender_urgency",
        "alpha", "eta", "Uw", "Us", "Y", "delay_proxy", "ACE_fd", "ACE_gc", "ACEstar", "ACEw_star", "ACEs_star",
        "region_density", "peak_flag",
    ]
    df = to_numeric_safe(df, numeric_candidates)

    required_core = ["task_id", "Uw", "Us", "Y", "ACEstar"]
    missing = [c for c in required_core if c not in df.columns]
    if missing:
        raise ValueError(
            "合并后缺少关键字段：" + ", ".join(missing) +
            f"\n请检查输入表字段名。当前字段：{list(df.columns)}"
        )

    risk_cols = ["semantic_difficulty", "environment_complexity", "path_uncertainty", "rejection_risk"]
    existing_risk_cols = [c for c in risk_cols if c in df.columns]
    if existing_risk_cols:
        for c in existing_risk_cols:
            df[f"{c}_norm"] = normalize_01(df[c])
        norm_cols = [f"{c}_norm" for c in existing_risk_cols]
        df["risk_mean"] = df[norm_cols].mean(axis=1)
    else:
        df["risk_mean"] = np.nan

    df["utility_gap"] = (df["Uw"] - df["Us"]).abs()

    preferred_cols = [
        "task_id",
        "reward", "distance_km", "time_window_sec", "task_difficulty_struct", "path_error",
        "region_density", "peak_flag",
        "semantic_difficulty", "environment_complexity", "path_uncertainty", "rejection_risk", "sender_urgency",
        "alpha", "eta",
        "Uw", "Us", "Y", "delay_proxy",
        "ACE_fd", "ACE_gc", "ACEstar", "ACEw_star", "ACEs_star",
        "risk_mean", "utility_gap",
    ]
    keep_cols = [c for c in preferred_cols if c in df.columns]
    evidence = df[keep_cols].copy()

    before = len(evidence)
    evidence = evidence.dropna(subset=["task_id", "Uw", "Us", "Y", "ACEstar"])
    after = len(evidence)
    if after == 0:
        raise ValueError("任务证据表为空，请检查输入数据。")

    evidence_path = out_dir / "task_ranking_evidence.csv"
    compact_path = out_dir / "task_ranking_evidence_compact.csv"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")

    compact_cols = [
        "task_id", "reward", "distance_km", "time_window_sec", "task_difficulty_struct",
        "path_uncertainty", "rejection_risk", "sender_urgency",
        "Uw", "Us", "Y", "ACEstar", "risk_mean", "utility_gap",
    ]
    compact_cols = [c for c in compact_cols if c in evidence.columns]
    round_numeric(evidence[compact_cols], digits=args.round_digits).to_csv(
        compact_path, index=False, encoding="utf-8-sig"
    )

    def safe_float(x):
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    summary = {
        "model": model,
        "num_tasks": int(after),
        "dropped_rows_due_to_missing_core_fields": int(before - after),
        "Y_mean": safe_float(evidence["Y"].mean()),
        "Uw_mean": safe_float(evidence["Uw"].mean()),
        "Us_mean": safe_float(evidence["Us"].mean()),
        "ACEstar_mean": safe_float(evidence["ACEstar"].mean()),
        "ACEstar_positive_ratio": safe_float((evidence["ACEstar"] > 0).mean()),
        "risk_mean": safe_float(evidence["risk_mean"].mean()) if "risk_mean" in evidence.columns else None,
        "risk_q70": safe_float(evidence["risk_mean"].quantile(0.70)) if "risk_mean" in evidence.columns else None,
        "utility_gap_mean": safe_float(evidence["utility_gap"].mean()),
        "utility_gap_q70": safe_float(evidence["utility_gap"].quantile(0.70)),
        "Y_q70": safe_float(evidence["Y"].quantile(0.70)),
        "ACEstar_near_zero_eps": safe_float(max(1e-8, 0.05 * evidence["ACEstar"].std()))
        if evidence["ACEstar"].std() == evidence["ACEstar"].std() else 1e-8,
    }
    save_json(summary, out_dir / "task_evidence_summary.json")

    initial_chain = {
        "chain_type": "global_initial_causal_ranking_chain",
        "model": model,
        "not_task_level_chain": True,
        "description": (
            "本文的初始因果排序思维链不是为每个任务单独生成一条解释链，"
            "而是面向整个候选任务集形成的初始排序推理逻辑。"
        ),
        "initial_chain_text": (
            "在众包物流任务推荐中，排序决策不能仅依据任务报酬、配送距离或时间窗等单一属性，"
            "而应同时考虑接包人侧收益负担和发包人侧完成效率。首先，任务报酬、配送距离、"
            "时间窗长度、路径差异和任务难度等属性会影响任务吸引力与执行压力；较高报酬通常有助于提高接包人接单意愿，"
            "较长距离、较短时间窗和较高任务难度则可能增加接包人执行成本。其次，任务文本和上下文信息会影响语义风险判断，"
            "语义难度、环境复杂度、路径不确定性、拒单风险和发包紧急度越高，任务越可能出现拒单、延迟接取或履约不稳定。"
            "最后，在候选任务排序时，应优先选择综合效用较高、接包人负担适中、发包人等待风险较低且整体风险可控的任务；"
            "对于收益不足、路径复杂、时间紧迫或拒单风险较高的任务，应降低其推荐优先级。"
        ),
        "input_evidence_fields": compact_cols,
    }
    save_json(initial_chain, out_dir / "global_initial_causal_ranking_chain.json")

    optimized_chain = {
        "chain_type": "global_optimized_causal_ranking_chain",
        "model": model,
        "not_task_level_chain": True,
        "causal_path": "X -> Z -> {Uw, Us, Y}",
        "description": (
            "该文件只保存一条面向候选任务集的全局优化因果排序思维链。"
            "task_ranking_evidence.csv 中的每一行是任务级因果证据，不是任务级思维链。"
        ),
        "optimized_chain_text": (
            "在对候选任务集进行公平排序时，大模型首先应按照 X -> Z -> {Uw, Us, Y} 的因果路径理解任务属性对双边效用的影响，"
            "不应仅依据任务报酬、配送距离或综合效用进行机械排序。其次，大模型应结合前门调整和 g-computation 结果判断任务属性变化"
            "是否真正具有双边效用改善空间，并将 ACEstar 作为任务进入优先候选的重要因果证据。若 ACEstar > 0，说明任务具有正向双边因果增益；"
            "若 ACEstar 接近 0，说明任务因果增益不明显，应保持稳健排序；若 ACEstar < 0，说明任务当前状态下不宜优先推荐。再次，"
            "大模型应比较 Uw、Us 和 Y，避免只对发包人或接包人单侧有利的任务被无条件排在前列。当 Uw 与 Us 均较优且差异较小时，"
            "说明任务更能兼顾双边公平；当 Y 较高但 Uw 与 Us 差异较大时，应标记为双边效用偏向并谨慎提升。最后，大模型应根据语义难度、"
            "路径不确定性和拒单风险对排序结果进行风险约束。即使 ACEstar 为正，若 risk_mean、path_uncertainty 或 rejection_risk 较高，"
            "也不应直接提升至前列。最终 Top-N 推荐任务应同时满足正向因果增益、较高综合效用、较好的双边效用均衡和较低风险水平。"
        ),
        "ranking_rules": [
            "优先考虑 ACEstar 为正的任务。",
            "在 ACEstar 为正的任务中，优先考虑 Y 较高的任务。",
            "不能只追求 Y 最大，还要比较 Uw 与 Us 的均衡性。",
            "若 utility_gap 较大，说明存在单侧效用偏向，应谨慎提升。",
            "若 risk_mean、path_uncertainty 或 rejection_risk 较高，即使 ACEstar 为正，也不能直接排在前列。",
            "最终 Top-N 应兼顾正向因果效应、双边均衡、综合效用和风险控制。",
        ],
        "action_rules": {
            "提升排序": "ACEstar > 0，Y 较高，Uw 与 Us 较均衡，risk_mean 较低。",
            "谨慎提升排序": "ACEstar > 0，但 risk_mean 较高或 utility_gap 较大。",
            "保持中等排序": "ACEstar 接近 0，综合效用一般但风险可控。",
            "降低排序": "ACEstar < 0，或风险较高且双边效用表现较差。",
        },
        "data_thresholds_for_reference": summary,
        "candidate_evidence_file": "task_ranking_evidence_compact.csv",
    }
    save_json(optimized_chain, out_dir / "global_optimized_causal_ranking_chain.json")

    print("=" * 80)
    print(f"[Step4 完成] model={model}")
    print(f"读取结构化特征：{struct_path}")
    print(f"读取语义变量：{llmz_path}")
    print(f"读取效用结果：{outcomes_path}")
    print(f"读取因果效应：{causal_path}")
    print(f"输出目录：{out_dir}")
    print(f"候选任务证据表：{evidence_path}，行数={after}")
    print(f"压缩证据表：{compact_path}")
    print("输出：global_initial_causal_ranking_chain.json")
    print("输出：global_optimized_causal_ranking_chain.json")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="实验根目录，例如 E:\\sim_experiment_data")
    parser.add_argument("--model", type=str, required=True, choices=["deepseek", "doubao", "gpt4o"], help="模型目录名")
    parser.add_argument("--step2_dir_name", type=str, default="step2_llmZ", help="Step2 输出目录名")
    parser.add_argument("--step3_dir_name", type=str, default="new_cot_fairrank", help="已有 Step3 因果效应目录名")
    parser.add_argument("--out_dir_name", type=str, default="global_chain_fairrank", help="Step4-6 新输出目录名")
    parser.add_argument("--round_digits", type=int, default=6, help="压缩证据表数值保留小数位")
    return parser.parse_args()


if __name__ == "__main__":
    build_step4_outputs(parse_args())