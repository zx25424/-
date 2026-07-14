# -*- coding: utf-8 -*-
r"""
Step5：用“全局优化因果排序思维链 + 候选任务级因果证据表”调用大模型生成 Top-N 排序。

核心逻辑：
1. 不再让大模型读取每个任务的 optimized_cot；
2. 只给大模型一条 global_optimized_causal_ranking_chain；
3. 同时输入 task_ranking_evidence_compact.csv 中的候选任务证据表；
4. 候选任务证据表采用 columns + rows 压缩格式，避免上下文过长；
5. 输出 TopN_llm_fair_ranking_global_chain.csv/json。

运行示例：
python step5_rank_with_global_chain.py --base_dir C:\Users\lenovo\Desktop\小论文1\new\pythonProject\sim_experiment_data --model deepseek --top_n 10 --max_tokens 1000

API 配置方式：
1）命令行传入：
   --api_key xxx --api_base xxx --llm_model xxx

2）环境变量：
   deepseek: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
   doubao:   DOUBAO_API_KEY,   DOUBAO_BASE_URL,   DOUBAO_MODEL
   gpt4o:    OPENAI_API_KEY,   OPENAI_BASE_URL,   OPENAI_MODEL

如果只是测试流程，不想调用 API：
python step5_rank_with_global_chain.py --base_dir ... --model deepseek --mock_rule_rank
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """兼容不同编码读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    """保存 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_for_score(s: pd.Series, higher_better: bool = True) -> pd.Series:
    """规则调试模式中使用的归一化函数。"""
    x = pd.to_numeric(s, errors="coerce").astype(float)

    mn, mx = x.min(), x.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        out = pd.Series(np.zeros(len(x)), index=x.index)
    else:
        out = (x - mn) / (mx - mn)

    return out if higher_better else 1 - out


def mock_rule_ranking(evidence: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    仅用于调试流程的规则排序，不建议作为正式实验替代大模型排序。
    正式实验请去掉 --mock_rule_rank，并配置 API。
    """
    df = evidence.copy()

    for c in ["ACEstar", "Y", "Uw", "Us", "risk_mean", "utility_gap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["score_ref"] = 0.0

    if "ACEstar" in df.columns:
        df["score_ref"] += 0.35 * normalize_for_score(df["ACEstar"], True)

    if "Y" in df.columns:
        df["score_ref"] += 0.30 * normalize_for_score(df["Y"], True)

    if "utility_gap" in df.columns:
        df["score_ref"] += 0.20 * normalize_for_score(df["utility_gap"], False)

    if "risk_mean" in df.columns:
        df["score_ref"] += 0.15 * normalize_for_score(df["risk_mean"], False)

    out = df.sort_values("score_ref", ascending=False).head(top_n).copy()

    rows = []

    utility_gap_q70 = df["utility_gap"].quantile(0.70) if "utility_gap" in df.columns else None
    ace_std = df["ACEstar"].std() if "ACEstar" in df.columns else 0.0
    ace_eps = max(1e-8, 0.05 * ace_std) if ace_std == ace_std else 1e-8

    for i, r in enumerate(out.to_dict("records"), start=1):
        ace = float(r.get("ACEstar", 0) or 0)
        risk = float(r.get("risk_mean", 0) or 0)
        gap = float(r.get("utility_gap", 0) or 0)

        if ace > 0 and risk < 0.7 and (utility_gap_q70 is None or gap <= utility_gap_q70):
            action = "提升排序"
        elif ace > 0:
            action = "谨慎提升排序"
        elif abs(ace) <= ace_eps:
            action = "保持中等排序"
        else:
            action = "降低排序"

        rows.append({
            "rank": i,
            "task_id": r.get("task_id"),
            "ranking_score": round(float(r.get("score_ref", 0)), 6),
            "ranking_reason": "规则调试模式：依据 ACEstar、Y、utility_gap、risk_mean 的参考得分排序。正式论文建议使用大模型输出。",
            "fairness_reason": f"Uw={r.get('Uw')}, Us={r.get('Us')}, utility_gap={r.get('utility_gap')}。",
            "key_causal_evidence": f"ACEstar={r.get('ACEstar')}, Y={r.get('Y')}。",
            "risk_warning": f"risk_mean={r.get('risk_mean')}, path_uncertainty={r.get('path_uncertainty')}, rejection_risk={r.get('rejection_risk')}。",
            "action": action,
            "global_chain_used": True,
        })

    return pd.DataFrame(rows)


def get_api_config(args: argparse.Namespace) -> Dict[str, str]:
    """按模型读取 API 配置。"""
    if args.api_key and args.api_base and args.llm_model:
        return {
            "api_key": args.api_key,
            "api_base": args.api_base,
            "llm_model": args.llm_model,
        }

    if args.model == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    elif args.model == "doubao":
        api_key = os.getenv("DOUBAO_API_KEY")
        api_base = os.getenv("DOUBAO_BASE_URL")
        llm_model = os.getenv("DOUBAO_MODEL")

    elif args.model == "gpt4o":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    else:
        raise ValueError(args.model)

    missing = []

    if not api_key:
        missing.append("api_key")

    if not api_base:
        missing.append("api_base")

    if not llm_model:
        missing.append("llm_model")

    if missing:
        raise ValueError(
            f"缺少 API 配置：{missing}。\n"
            "可以通过命令行 --api_key --api_base --llm_model 传入，"
            "或设置对应环境变量。若只测试流程，可添加 --mock_rule_rank。"
        )

    return {
        "api_key": api_key,
        "api_base": api_base,
        "llm_model": llm_model,
    }


def evidence_to_compact_json(evidence: pd.DataFrame, max_rows: int = 0) -> str:
    """
    将候选任务证据表压缩为 columns + rows 格式，避免每一行重复字段名。

    max_rows=0 表示全部候选任务。
    正式实验建议 max_rows=0，即保留 1450 个候选任务，只压缩字段和格式。
    """
    df = evidence.copy()

    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()

    # 只保留排序最关键字段，减少上下文长度。
    # 注意：这不会筛掉任务，只是减少每个任务输入给大模型的字段数量。
    preferred_cols = [
        "task_id",
        "Uw",
        "Us",
        "Y",
        "ACEstar",
        "risk_mean",
        "utility_gap",
        "path_uncertainty",
        "rejection_risk",
    ]

    cols = [c for c in preferred_cols if c in df.columns]

    if "task_id" not in cols:
        raise ValueError("候选任务证据表中缺少 task_id 字段。")

    required = ["Uw", "Us", "Y", "ACEstar"]
    missing_required = [c for c in required if c not in cols]

    if missing_required:
        raise ValueError(
            "候选任务证据表缺少 Step5 排序所需关键字段："
            + ", ".join(missing_required)
            + f"\n当前字段：{list(df.columns)}"
        )

    df = df[cols].copy()

    # 数值保留 5 位小数，减少 token。
    for c in cols:
        if c != "task_id":
            df[c] = pd.to_numeric(df[c], errors="coerce").round(5)

    rows = []

    for row in df.to_dict("records"):
        item = []

        for c in cols:
            v = row.get(c)

            if pd.isna(v):
                item.append(None)
            elif isinstance(v, np.integer):
                item.append(int(v))
            elif isinstance(v, (np.floating, float)):
                item.append(round(float(v), 5))
            else:
                item.append(v)

        rows.append(item)

    compact_obj = {
        "format": "columns_rows",
        "note": "columns 为字段名，rows 中每一行的数值顺序与 columns 一一对应。每一行表示一个候选任务。",
        "columns": cols,
        "rows": rows,
    }

    return json.dumps(compact_obj, ensure_ascii=False, separators=(",", ":"))


def build_prompt(chain: dict, evidence_json: str, top_n: int, num_tasks: int) -> str:
    """构造大模型排序 Prompt。"""
    chain_text = chain.get("optimized_chain_text") or json.dumps(chain, ensure_ascii=False)
    rules = chain.get("ranking_rules", [])
    action_rules = chain.get("action_rules", {})

    prompt = f"""
你是众包物流双边公平推荐排序模型。现在给你一条【全局优化因果排序思维链】以及候选任务集的【任务级因果证据表】。

请特别注意：
1. 这里只有一条全局优化因果排序思维链；
2. 候选任务表中的每一行是一个任务的证据，不是每个任务一条思维链；
3. 你的任务是根据这条全局排序思维链，对候选任务集直接比较并输出 Top-{top_n}；
4. 最终 task_id 必须来自候选任务证据表，不能编造。

【全局优化因果排序思维链】
{chain_text}

【排序规则】
{json.dumps(rules, ensure_ascii=False, indent=2)}

【排序动作规则】
{json.dumps(action_rules, ensure_ascii=False, indent=2)}

【候选任务数量】
{num_tasks}

【候选任务级因果证据表】
下面的候选任务表采用 columns_rows 格式：
- columns 表示字段名；
- rows 中每一行表示一个候选任务；
- rows 每一行的数值顺序与 columns 完全对应；
- task_id 是任务编号，最终输出的 task_id 必须来自该表。

{evidence_json}

请基于上述全局优化因果排序思维链，对候选任务集进行直接比较，并输出 Top-{top_n} 公平推荐结果。

必须遵守：
1. 优先考虑 ACEstar 为正的任务；
2. 在 ACEstar 为正的任务中，优先考虑 Y 较高的任务；
3. 不能只追求 Y 最大，还要比较 Uw 和 Us 的均衡性，utility_gap 过大时应谨慎；
4. risk_mean、path_uncertainty 或 rejection_risk 较高时，即使 ACEstar 为正，也不能无条件排在前列；
5. Top-{top_n} 应兼顾正向因果效应、双边效用均衡、综合效用和风险控制；
6. task_id 必须来自候选任务表，不能编造；
7. 严格返回 JSON 对象，不要返回 Markdown，不要添加解释性正文；
8. action 只能从以下四类中选择：提升排序、谨慎提升排序、保持中等排序、降低排序。

JSON 格式如下：
{{
  "top10": [
    {{
      "rank": 1,
      "task_id": "候选任务中的 task_id",
      "ranking_score": 0.0,
      "ranking_reason": "排序依据",
      "fairness_reason": "双边公平依据，必须提到 Uw、Us 或 utility_gap",
      "key_causal_evidence": "关键因果证据，必须提到 ACEstar 或因果增益",
      "risk_warning": "风险提示，必须提到 risk_mean、path_uncertainty 或 rejection_risk",
      "action": "提升排序/谨慎提升排序/保持中等排序/降低排序",
      "global_chain_used": true
    }}
  ]
}}
""".strip()

    return prompt


def extract_json_object(text: str) -> dict:
    """从模型输出中解析 JSON 对象。"""
    raw = text.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:
        return json.loads(raw[start: end + 1])

    raise ValueError("无法从模型输出中解析 JSON。原始输出前 1000 字：\n" + raw[:1000])


def call_openai_compatible_api(args: argparse.Namespace, prompt: str) -> str:
    """调用 OpenAI 兼容接口。"""
    cfg = get_api_config(args)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("请先安装 openai：pip install openai") from e

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["api_base"],
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg["llm_model"],
                messages=[
                    {
                        "role": "system",
                        "content": "你是严格输出 JSON 的众包物流双边公平推荐排序模型。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            return resp.choices[0].message.content or ""

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** attempt)
            print(f"[API 调用失败] attempt={attempt}/{args.max_retries}, 等待 {wait}s, error={e}")
            time.sleep(wait)

    raise RuntimeError(f"API 调用失败，已重试 {args.max_retries} 次：{last_error}")


def validate_topn(result_df: pd.DataFrame, evidence: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """检查 task_id 是否来自候选集，并补齐字段。"""
    if "task_id" not in result_df.columns:
        raise ValueError("模型输出缺少 task_id 字段。")

    valid_task_ids = set(evidence["task_id"].astype(str))
    result_df["task_id"] = result_df["task_id"].astype(str)

    invalid = [tid for tid in result_df["task_id"].tolist() if tid not in valid_task_ids]

    if invalid:
        raise ValueError(f"模型输出包含候选集中不存在的 task_id：{invalid[:20]}")

    result_df = result_df.drop_duplicates(subset=["task_id"], keep="first").copy()

    if "rank" in result_df.columns:
        result_df["rank"] = pd.to_numeric(result_df["rank"], errors="coerce")
        result_df = result_df.sort_values("rank")

    result_df = result_df.head(top_n).copy()
    result_df["rank"] = list(range(1, len(result_df) + 1))

    default_cols = {
        "ranking_score": np.nan,
        "ranking_reason": "",
        "fairness_reason": "",
        "key_causal_evidence": "",
        "risk_warning": "",
        "action": "保持中等排序",
        "global_chain_used": True,
    }

    for c, v in default_cols.items():
        if c not in result_df.columns:
            result_df[c] = v

    result_df["global_chain_used"] = result_df["global_chain_used"].fillna(True)

    return result_df


def run_step5(args: argparse.Namespace) -> None:
    """Step5 主函数。"""
    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.model / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_path = out_dir / "global_optimized_causal_ranking_chain.json"
    evidence_path = out_dir / "task_ranking_evidence_compact.csv"

    if not evidence_path.exists():
        evidence_path = out_dir / "task_ranking_evidence.csv"

    chain = load_json(chain_path)
    evidence = read_csv_flexible(evidence_path)

    if "task_id" not in evidence.columns:
        raise ValueError(f"{evidence_path} 中缺少 task_id 字段。")

    evidence["task_id"] = evidence["task_id"].astype(str)

    if args.mock_rule_rank:
        topn_df = mock_rule_ranking(evidence, args.top_n)
        raw_response = {
            "mock_rule_rank": True,
            "note": "规则调试模式，不是大模型输出。",
        }

    else:
        evidence_json = evidence_to_compact_json(
            evidence,
            max_rows=args.max_candidates,
        )

        num_tasks_sent = len(evidence) if args.max_candidates == 0 else min(args.max_candidates, len(evidence))

        prompt = build_prompt(
            chain=chain,
            evidence_json=evidence_json,
            top_n=args.top_n,
            num_tasks=num_tasks_sent,
        )

        prompt_path = out_dir / "llm_global_chain_ranking_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        raw_text = call_openai_compatible_api(args, prompt)

        raw_response_path = out_dir / "llm_global_chain_ranking_raw_response.txt"
        raw_response_path.write_text(raw_text, encoding="utf-8")

        parsed = extract_json_object(raw_text)
        raw_response = parsed

        if "top10" in parsed:
            records = parsed["top10"]
        elif "topn" in parsed:
            records = parsed["topn"]
        elif "results" in parsed:
            records = parsed["results"]
        else:
            raise ValueError(f"模型 JSON 中找不到 top10/topn/results 字段：{parsed.keys()}")

        topn_df = pd.DataFrame(records)

    topn_df = validate_topn(topn_df, evidence, args.top_n)
    topn_df["model_name"] = args.model

    csv_path = out_dir / "TopN_llm_fair_ranking_global_chain.csv"
    json_path = out_dir / "TopN_llm_fair_ranking_global_chain.json"
    log_path = out_dir / "llm_global_chain_ranking_log.json"

    topn_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_json(topn_df.to_dict("records"), json_path)

    save_json(
        {
            "model": args.model,
            "top_n": args.top_n,
            "num_candidates_in_file": int(len(evidence)),
            "num_candidates_sent_to_llm": int(
                len(evidence) if args.max_candidates == 0 else min(args.max_candidates, len(evidence))
            ),
            "mock_rule_rank": bool(args.mock_rule_rank),
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "raw_response": raw_response,
        },
        log_path,
    )

    print("=" * 80)
    print(f"[Step5 完成] model={args.model}")
    print(f"输入全局排序链：{chain_path}")
    print(f"输入任务证据表：{evidence_path}，候选任务数={len(evidence)}")
    print(f"输出 TopN CSV：{csv_path}")
    print(f"输出 TopN JSON：{json_path}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="实验根目录，例如 C:\\Users\\lenovo\\Desktop\\小论文1\\new\\pythonProject\\sim_experiment_data",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["deepseek", "doubao", "gpt4o"],
        help="模型目录名",
    )

    parser.add_argument(
        "--out_dir_name",
        type=str,
        default="global_chain_fairrank",
        help="Step4-6 输出目录名",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="输出 Top-N 推荐结果，默认 10",
    )

    parser.add_argument(
        "--max_candidates",
        type=int,
        default=0,
        help="传给大模型的最大候选数；0 表示全部候选任务。正式实验建议 0。",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="大模型温度参数",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="模型输出最大 token。为避免上下文超限，默认改为 1000。",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="API 调用最大重试次数",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API Key。也可使用环境变量。",
    )

    parser.add_argument(
        "--api_base",
        type=str,
        default="",
        help="OpenAI 兼容接口 base_url。",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="",
        help="大模型名称或接入点 ID。",
    )

    parser.add_argument(
        "--mock_rule_rank",
        action="store_true",
        help="仅用于调试流程，不调用大模型。正式实验不要使用。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_step5(parse_args())