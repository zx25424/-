# step7_plot_new_framework_results.py
# -*- coding: utf-8 -*-

import os
import ast
import textwrap
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = "sim_experiment_data"
OUT_FIG_DIR = os.path.join(BASE_DIR, "new_framework_figures")
os.makedirs(OUT_FIG_DIR, exist_ok=True)

MODEL_DIRS = {
    "DeepSeek": os.path.join(BASE_DIR, "deepseek", "new_cot_fairrank"),
    "Doubao": os.path.join(BASE_DIR, "doubao", "new_cot_fairrank"),
    "GPT-4o": os.path.join(BASE_DIR, "gpt4o", "new_cot_fairrank"),
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 工具函数
# ============================================================

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_dict_str(x):
    """
    把 evaluation_summary.csv 里的字符串字典转为 dict
    """
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return {}


def wrap_text(x, width=18):
    if pd.isna(x):
        return ""
    return "\n".join(textwrap.wrap(str(x), width=width))


def load_summary_df():
    """
    读取三个模型的 evaluation_summary.csv，并合并成一个总表
    """
    rows = []

    for model_name, model_dir in MODEL_DIRS.items():
        summary_path = os.path.join(
            model_dir, "step6_eval", "evaluation_summary.csv"
        )
        df = safe_read_csv(summary_path)
        row = df.iloc[0].to_dict()
        row["model"] = model_name
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def load_topn_df(model_name):
    """
    读取某个模型的 Top-N 排序结果
    """
    model_dir = MODEL_DIRS[model_name]
    topn_path = os.path.join(
        model_dir, "step5_llm_ranking", "TopN_llm_fair_ranking.csv"
    )
    return safe_read_csv(topn_path)


def load_detail_df(model_name):
    """
    读取某个模型的带指标明细结果
    """
    model_dir = MODEL_DIRS[model_name]
    detail_path = os.path.join(
        model_dir, "step6_eval", "ranking_with_metrics.csv"
    )
    return safe_read_csv(detail_path)


def save_bar_chart_from_df(
    df_plot,
    title,
    xlabel,
    ylabel,
    save_path,
    rotation=0
):
    """
    通用柱状图保存函数
    """
    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. 图1：最终 Top-N 推荐结果图（每个模型各一张）
# ============================================================

def plot_topn_result_figures():
    for model_name in MODEL_DIRS.keys():
        df = load_topn_df(model_name).copy()

        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
            df = df.sort_values("rank").head(10)
        else:
            df = df.head(10)

        df["ranking_score"] = pd.to_numeric(df["ranking_score"], errors="coerce").fillna(0)

        labels = []
        for _, row in df.iterrows():
            rank = int(row["rank"]) if "rank" in df.columns and pd.notna(row["rank"]) else ""
            task_id = row["task_id"] if "task_id" in df.columns else ""
            labels.append(f"R{rank}\nT{task_id}")

        plt.figure(figsize=(12, 6))
        plt.bar(labels, df["ranking_score"])
        plt.title(f"{model_name} 最终 Top-N 推荐结果图")
        plt.xlabel("推荐顺位 / 任务ID")
        plt.ylabel("ranking_score")

        if "action" in df.columns:
            for i, (_, row) in enumerate(df.iterrows()):
                score = row["ranking_score"]
                action = str(row["action"])
                plt.text(i, score, action, ha="center", va="bottom", fontsize=8, rotation=90)

        plt.tight_layout()
        save_path = os.path.join(OUT_FIG_DIR, f"fig_topn_result_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("[Saved]", save_path)


# ============================================================
# 4. 图2：三模型公平性对比图
# ============================================================

def plot_fairness_comparison(summary_df):
    # 4.1 双边效用均值对比
    cols1 = ["Uw_mean", "Us_mean", "Y_mean"]
    df1 = summary_df[["model"] + cols1].set_index("model")

    plt.figure(figsize=(10, 6))
    ax = df1.plot(kind="bar", figsize=(10, 6))
    ax.set_title("三模型双边效用对比图")
    ax.set_xlabel("模型")
    ax.set_ylabel("指标值")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path1 = os.path.join(OUT_FIG_DIR, "fig_fairness_utility_comparison.png")
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path1)

    # 4.2 双边公平核心指标对比
    cols2 = ["bilateral_gap_mean", "bilateral_balance_score"]
    df2 = summary_df[["model"] + cols2].set_index("model")

    plt.figure(figsize=(10, 6))
    ax = df2.plot(kind="bar", figsize=(10, 6))
    ax.set_title("三模型双边公平性对比图")
    ax.set_xlabel("模型")
    ax.set_ylabel("指标值")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path2 = os.path.join(OUT_FIG_DIR, "fig_fairness_core_comparison.png")
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path2)


# ============================================================
# 5. 图3：三模型风险控制图
# ============================================================

def plot_risk_comparison(summary_df):
    cols = ["risk_mean", "high_risk_ratio"]
    df_plot = summary_df[["model"] + cols].set_index("model")

    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("三模型风险控制对比图")
    ax.set_xlabel("模型")
    ax.set_ylabel("指标值")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(OUT_FIG_DIR, "fig_risk_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path)


# ============================================================
# 6. 图4：优化因果思维链质量对比图
# ============================================================

def plot_cot_quality_comparison(summary_df):
    cols = ["optimized_cot_non_empty_ratio", "cot_confidence_mean"]
    df_plot = summary_df[["model"] + cols].set_index("model")

    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("三模型优化因果思维链质量对比图")
    ax.set_xlabel("模型")
    ax.set_ylabel("指标值")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(OUT_FIG_DIR, "fig_cot_quality_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path)


# ============================================================
# 7. 图5：优化因果思维链动作分布图（每个模型各一张）
# ============================================================

def plot_action_distribution_figures():
    for model_name in MODEL_DIRS.keys():
        df = load_topn_df(model_name).copy()

        if "action" not in df.columns:
            continue

        counts = df["action"].fillna("未说明").value_counts()

        plt.figure(figsize=(10, 6))
        counts.plot(kind="bar")
        plt.title(f"{model_name} 优化因果思维链动作分布图")
        plt.xlabel("动作类型")
        plt.ylabel("数量")
        plt.xticks(rotation=20)
        plt.tight_layout()
        save_path = os.path.join(OUT_FIG_DIR, f"fig_action_distribution_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("[Saved]", save_path)


# ============================================================
# 8. 图6：优化因果思维链解释案例表（每个模型各一张）
# ============================================================

def plot_cot_case_table_figures(top_k=3):
    for model_name in MODEL_DIRS.keys():
        df = load_detail_df(model_name).copy()

        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
            df = df.sort_values("rank").head(top_k)
        else:
            df = df.head(top_k)

        show_cols = [
            "rank",
            "task_id",
            "action",
            "dominant_fairness_goal",
            "key_causal_path",
            "optimized_cot"
        ]

        show_cols = [c for c in show_cols if c in df.columns]
        table_df = df[show_cols].copy()

        for col in table_df.columns:
            if col == "optimized_cot":
                table_df[col] = table_df[col].apply(lambda x: wrap_text(x, width=28))
            elif col == "key_causal_path":
                table_df[col] = table_df[col].apply(lambda x: wrap_text(x, width=18))
            elif col == "dominant_fairness_goal":
                table_df[col] = table_df[col].apply(lambda x: wrap_text(x, width=12))
            else:
                table_df[col] = table_df[col].apply(lambda x: wrap_text(x, width=10))

        fig_height = 2 + top_k * 1.6
        plt.figure(figsize=(18, fig_height))
        ax = plt.gca()
        ax.axis("off")
        ax.set_title(f"{model_name} 优化因果思维链解释案例表", fontsize=14, pad=12)

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="left",
            colLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)

        # 调整列宽
        col_widths = {
            0: 0.06,   # rank
            1: 0.08,   # task_id
            2: 0.12,   # action
            3: 0.15,   # dominant_fairness_goal
            4: 0.20,   # key_causal_path
            5: 0.39    # optimized_cot
        }

        for (row, col), cell in table.get_celld().items():
            if col in col_widths:
                cell.set_width(col_widths[col])

        plt.tight_layout()
        save_path = os.path.join(OUT_FIG_DIR, f"fig_cot_case_table_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("[Saved]", save_path)


# ============================================================
# 9. 图7：三模型 Top-N 排序动作对比图
# ============================================================

def plot_action_comparison_across_models():
    rows = []

    for model_name in MODEL_DIRS.keys():
        df = load_topn_df(model_name).copy()
        if "action" not in df.columns:
            continue

        counts = df["action"].fillna("未说明").value_counts().to_dict()

        row = {"model": model_name}
        for k, v in counts.items():
            row[k] = v
        rows.append(row)

    if not rows:
        return

    action_df = pd.DataFrame(rows).fillna(0)
    action_df = action_df.set_index("model")

    plt.figure(figsize=(12, 6))
    ax = action_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("三模型 Top-N 排序动作对比图")
    ax.set_xlabel("模型")
    ax.set_ylabel("数量")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(OUT_FIG_DIR, "fig_action_comparison_across_models.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[Saved]", save_path)


# ============================================================
# 10. 主程序
# ============================================================

def main():
    print("开始绘制新实验框架图表...")

    summary_df = load_summary_df()

    # 保存一个汇总表，方便论文表格使用
    summary_out = os.path.join(OUT_FIG_DIR, "summary_all_models.csv")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print("[Saved]", summary_out)

    # 图1：最终 Top-N 推荐结果图
    plot_topn_result_figures()

    # 图2：三模型公平性对比图
    plot_fairness_comparison(summary_df)

    # 图3：三模型风险控制图
    plot_risk_comparison(summary_df)

    # 图4：优化因果思维链质量对比图
    plot_cot_quality_comparison(summary_df)

    # 图5：各模型动作分布图
    plot_action_distribution_figures()

    # 图6：优化因果思维链解释案例表
    plot_cot_case_table_figures(top_k=3)

    # 图7：三模型动作对比图
    plot_action_comparison_across_models()

    print("\n全部图表生成完成。")
    print("输出目录：", OUT_FIG_DIR)


if __name__ == "__main__":
    main()