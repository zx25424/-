# -*- coding: utf-8 -*-
"""
Step6：评价“全局优化因果排序思维链约束下”的 Top-N 推荐结果。

输入：
- TopN_llm_fair_ranking_global_chain.csv
- task_ranking_evidence.csv

输出：
- TopN_llm_fair_ranking_global_chain_with_evidence.csv
- evaluation_metrics_global_chain.json
- evaluation_metrics_global_chain.csv

运行示例：
python step6_eval_global_chain.py --base_dir E:\sim_experiment_data --model deepseek
python step6_eval_global_chain.py --base_dir E:\sim_experiment_data --model doubao
python step6_eval_global_chain.py --base_dir E:\sim_experiment_data --model gpt4o
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Any:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def ratio_non_empty(s: pd.Series) -> float:
    return float(s.fillna("").astype(str).str.strip().str.len().gt(0).mean())


def evaluate_one_model(args: argparse.Namespace) -> Dict[str, Any]:
    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.model / args.out_dir_name

    topn_path = out_dir / "TopN_llm_fair_ranking_global_chain.csv"
    evidence_path = out_dir / "task_ranking_evidence.csv"
    compact_path = out_dir / "task_ranking_evidence_compact.csv"

    topn = read_csv_flexible(topn_path)
    evidence = read_csv_flexible(evidence_path if evidence_path.exists() else compact_path)

    if "task_id" not in topn.columns or "task_id" not in evidence.columns:
        raise ValueError("TopN 或 evidence 表缺少 task_id 字段。")

    topn["task_id"] = topn["task_id"].astype(str)
    evidence["task_id"] = evidence["task_id"].astype(str)

    merged = topn.merge(evidence, on="task_id", how="left", suffixes=("", "_evidence"))

    for c in ["Uw", "Us", "Y", "ACEstar", "risk_mean", "utility_gap", "ranking_score"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    metrics: Dict[str, Any] = {}
    metrics["model"] = args.model
    metrics["num_topn"] = int(len(merged))
    metrics["selected_task_ids"] = merged["task_id"].tolist()

    for col in ["Uw", "Us", "Y", "ACEstar", "risk_mean", "utility_gap", "ranking_score"]:
        if col in merged.columns:
            metrics[f"{col}_mean"] = safe_float(merged[col].mean())
            metrics[f"{col}_std"] = safe_float(merged[col].std())
            metrics[f"{col}_min"] = safe_float(merged[col].min())
            metrics[f"{col}_max"] = safe_float(merged[col].max())

    if "Uw" in merged.columns and "Us" in merged.columns:
        gap = (merged["Uw"] - merged["Us"]).abs()
        metrics["bilateral_gap_mean"] = safe_float(gap.mean())
        metrics["bilateral_gap_std"] = safe_float(gap.std())
        metrics["bilateral_balance_score"] = safe_float(1.0 / (1.0 + gap.mean()))

    if "ACEstar" in merged.columns:
        metrics["positive_ACEstar_ratio"] = safe_float((merged["ACEstar"] > 0).mean())
        eps = max(1e-8, 0.05 * merged["ACEstar"].std()) if merged["ACEstar"].std() == merged["ACEstar"].std() else 1e-8
        metrics["near_zero_ACEstar_ratio"] = safe_float((merged["ACEstar"].abs() <= eps).mean())
        metrics["negative_ACEstar_ratio"] = safe_float((merged["ACEstar"] < 0).mean())

    if "risk_mean" in merged.columns:
        if "risk_mean" in evidence.columns:
            risk_threshold = pd.to_numeric(evidence["risk_mean"], errors="coerce").quantile(args.high_risk_quantile)
        else:
            risk_threshold = args.high_risk_fixed_threshold

        if pd.isna(risk_threshold):
            risk_threshold = args.high_risk_fixed_threshold

        metrics["high_risk_threshold"] = safe_float(risk_threshold)
        metrics["high_risk_ratio"] = safe_float((merged["risk_mean"] >= risk_threshold).mean())

    if "action" in merged.columns:
        metrics["action_distribution"] = merged["action"].fillna("未知").value_counts().to_dict()

    if "global_chain_used" in merged.columns:
        s = merged["global_chain_used"].astype(str).str.lower().str.strip()
        metrics["global_chain_used_ratio"] = float(s.isin(["true", "1", "yes", "是"]).mean())
    else:
        metrics["global_chain_used_ratio"] = 1.0

    for col in ["ranking_reason", "fairness_reason", "key_causal_evidence", "risk_warning"]:
        if col in merged.columns:
            metrics[f"{col}_non_empty_ratio"] = ratio_non_empty(merged[col])

    evidence_required = [c for c in ["Uw", "Us", "Y", "ACEstar", "risk_mean", "utility_gap"] if c in merged.columns]
    if evidence_required:
        metrics["task_evidence_complete_ratio"] = float(merged[evidence_required].notna().all(axis=1).mean())

    merged_path = out_dir / "TopN_llm_fair_ranking_global_chain_with_evidence.csv"
    metrics_json_path = out_dir / "evaluation_metrics_global_chain.json"
    metrics_csv_path = out_dir / "evaluation_metrics_global_chain.csv"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    save_json(metrics, metrics_json_path)
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"[Step6 完成] model={args.model}")
    print(f"输入 TopN：{topn_path}")
    print(f"输入证据表：{evidence_path if evidence_path.exists() else compact_path}")
    print(f"输出合并结果：{merged_path}")
    print(f"输出指标 JSON：{metrics_json_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("=" * 80)

    return metrics


def aggregate_if_needed(args: argparse.Namespace) -> None:
    """可选：汇总三个模型的 evaluation_metrics_global_chain.json。"""
    if not args.aggregate:
        return

    base_dir = Path(args.base_dir)
    rows: List[Dict[str, Any]] = []

    for model in ["deepseek", "doubao", "gpt4o"]:
        p = base_dir / model / args.out_dir_name / "evaluation_metrics_global_chain.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))

    if rows:
        out_path = base_dir / "common" / "evaluation_metrics_global_chain_all_models.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[三模型汇总完成] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="实验根目录，例如 E:\\sim_experiment_data")
    parser.add_argument("--model", type=str, required=True, choices=["deepseek", "doubao", "gpt4o"], help="模型目录名")
    parser.add_argument("--out_dir_name", type=str, default="global_chain_fairrank", help="Step4-6 输出目录名")
    parser.add_argument("--high_risk_quantile", type=float, default=0.70, help="用全候选集风险分位数定义高风险阈值")
    parser.add_argument("--high_risk_fixed_threshold", type=float, default=0.70, help="无法计算分位数时使用的风险阈值")
    parser.add_argument("--aggregate", action="store_true", help="运行后额外汇总已有三模型指标")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    evaluate_one_model(parsed_args)
    aggregate_if_needed(parsed_args)