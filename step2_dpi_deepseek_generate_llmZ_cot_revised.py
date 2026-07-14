# -*- coding: utf-8 -*-
"""
step2_dpi_deepseek_generate_llmZ_cot_revised.py

基于你之前可运行的 DeepSeek 版本重写：
1. 保留旧版稳定的 API 调用与 JSON 提取逻辑
2. 新增 causal_cot 输出
3. 新增论文两张表自动生成
4. 保留断点续跑、日志审计、QC 抽检

运行前 PowerShell 环境变量示例：
$env:DEEPSEEK_API_KEY="你的key"
$env:DATA_DIR="./sim_experiment_data"
$env:OUT_DIR="./sim_experiment_data"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:DPI_TIMEOUT="60"
$env:DPI_MAX_RETRY="2"
$env:DPI_TEMPERATURE="0.2"
$env:QC_SAMPLE_N="200"
"""

import os
import json
import time
import random
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests


# ===================== 配置 =====================
DATA_DIR = os.getenv("DATA_DIR", "sim_experiment_data")
OUT_DIR = os.getenv("OUT_DIR", DATA_DIR)

TASK_FILE = os.path.join(DATA_DIR, "T_task_raw.csv")
STRUCT_FILE = os.path.join(DATA_DIR, "T_features_struct.csv")

OUT_LLMZ = os.path.join(OUT_DIR, "T_features_llmZ_deepseek.csv")
OUT_LOG = os.path.join(OUT_DIR, "T_dpi_log_deepseek.csv")
OUT_QC = os.path.join(OUT_DIR, "QC_report_deepseek.json")
OUT_QC_SAMPLE = os.path.join(OUT_DIR, "QC_sample_200_deepseek.csv")

OUT_CASE_TABLE = os.path.join(OUT_DIR, "table_cot_cases_deepseek.csv")
OUT_SUMMARY_TABLE = os.path.join(OUT_DIR, "table_cot_summary_deepseek.csv")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

TEMPERATURE = float(os.getenv("DPI_TEMPERATURE", "0.2"))
TIMEOUT_S = int(os.getenv("DPI_TIMEOUT", "60"))
MAX_RETRY = int(os.getenv("DPI_MAX_RETRY", "2"))

QC_SAMPLE_N = int(os.getenv("QC_SAMPLE_N", "200"))
QC_SEED = int(os.getenv("QC_SEED", "0"))

random.seed(QC_SEED)
np.random.seed(QC_SEED)


# ===================== 工具函数 =====================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_alpha_eta(alpha: float, eta: float) -> Tuple[float, float]:
    alpha = clip01(alpha)
    eta = clip01(eta)
    s = alpha + eta
    if s <= 0:
        return 0.5, 0.5
    return alpha / s, eta / s


def extract_json_from_content(content: str):
    """
    兼容：
    - 纯 JSON
    - ```json ... ``` 包裹
    - ``` ... ``` 包裹
    - 前后有解释文字
    """
    if content is None:
        raise ValueError("empty_content")

    s = content.strip()

    # fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    # 截取第一个 JSON 块
    m2 = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
    if m2:
        s = m2.group(1).strip()

    return json.loads(s)


def append_csv(path: str, df: pd.DataFrame):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def load_existing_llmz():
    if os.path.exists(OUT_LLMZ):
        df = pd.read_csv(OUT_LLMZ)
        if "task_id" in df.columns:
            df["task_id"] = df["task_id"].astype(str)
        return df
    return pd.DataFrame()


def load_existing_log():
    if os.path.exists(OUT_LOG):
        df = pd.read_csv(OUT_LOG)
        if "task_id" in df.columns:
            df["task_id"] = df["task_id"].astype(str)
        return df
    return pd.DataFrame()


# ===================== Prompt（新增 causal_cot） =====================
SYSTEM_PROMPT = (
    "你是众包物流平台实验的“特征生成器”。"
    "你必须输出严格 JSON（只能输出 JSON，不允许任何多余文本、解释、Markdown、代码块）。"
    "JSON 必须包含以下字段："
    "semantic_difficulty, environment_complexity, path_uncertainty, rejection_risk, sender_urgency, "
    "alpha, eta, rationale, causal_cot。"
    "其中 semantic_difficulty/environment_complexity/path_uncertainty/rejection_risk/"
    "sender_urgency/alpha/eta 都必须是 [0,1] 的数字。"
    "alpha + eta 必须约等于 1。"
    "rationale 是一句中文简短解释，不超过 60 字。"
    "causal_cot 必须是对象，包含："
    "dominant_side, key_causal_path, secondary_path, weight_reason, decision_rule, cot_score_confidence。"
    "其中 dominant_side 只能是 sender 或 worker；"
    "decision_rule 只能是 eta > alpha、alpha > eta、eta ≈ alpha；"
    "cot_score_confidence 必须是 [0,1] 的数字；"
    "key_causal_path、secondary_path、weight_reason 都是不超过 50 字的中文短句。"
)


def build_user_prompt(payload: dict) -> str:
    return "输入任务信息（JSON）如下：\n" + json.dumps(payload, ensure_ascii=False)


# ===================== DeepSeek API 调用 =====================
def validate_output(out: Dict[str, Any]) -> Dict[str, Any]:
    must = [
        "semantic_difficulty", "environment_complexity", "path_uncertainty",
        "rejection_risk", "sender_urgency", "alpha", "eta", "rationale", "causal_cot"
    ]
    for k in must:
        if k not in out:
            raise ValueError(f"missing_field:{k}")

    clean = {}
    for k in [
        "semantic_difficulty", "environment_complexity", "path_uncertainty",
        "rejection_risk", "sender_urgency", "alpha", "eta"
    ]:
        v = float(out[k])
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"out_of_range:{k}:{v}")
        clean[k] = clip01(v)

    clean["alpha"], clean["eta"] = normalize_alpha_eta(clean["alpha"], clean["eta"])
    clean["rationale"] = str(out["rationale"]).strip()

    cot = out["causal_cot"]
    if not isinstance(cot, dict):
        raise ValueError("causal_cot_not_object")

    cot_required = [
        "dominant_side", "key_causal_path", "secondary_path",
        "weight_reason", "decision_rule", "cot_score_confidence"
    ]
    for k in cot_required:
        if k not in cot:
            raise ValueError(f"missing_cot_field:{k}")

    dominant_side = str(cot["dominant_side"]).strip()
    if dominant_side not in ["sender", "worker"]:
        raise ValueError(f"invalid_dominant_side:{dominant_side}")

    decision_rule = str(cot["decision_rule"]).strip()
    if decision_rule not in ["eta > alpha", "alpha > eta", "eta ≈ alpha"]:
        raise ValueError(f"invalid_decision_rule:{decision_rule}")

    cot_conf = float(cot["cot_score_confidence"])
    if not (0.0 <= cot_conf <= 1.0):
        raise ValueError(f"cot_score_confidence_out_of_range:{cot_conf}")

    clean["causal_cot"] = {
        "dominant_side": dominant_side,
        "key_causal_path": str(cot["key_causal_path"]).strip(),
        "secondary_path": str(cot["secondary_path"]).strip(),
        "weight_reason": str(cot["weight_reason"]).strip(),
        "decision_rule": decision_rule,
        "cot_score_confidence": clip01(cot_conf),
    }

    # 轻微一致性修正
    alpha, eta = clean["alpha"], clean["eta"]
    if dominant_side == "sender" and eta < alpha:
        clean["alpha"], clean["eta"] = normalize_alpha_eta(alpha * 0.8, eta * 1.2)
    elif dominant_side == "worker" and alpha < eta:
        clean["alpha"], clean["eta"] = normalize_alpha_eta(alpha * 1.2, eta * 0.8)

    return clean


def call_deepseek_strict_json(payload: dict):
    """
    返回：
      (ok:bool, out_json:dict|None, raw_text:str|None, meta:dict)
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，请用环境变量注入。")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(payload)}
        ],
        "temperature": TEMPERATURE
    }

    start = time.time()
    last_err = None
    attempts = 0
    last_raw = ""

    for attempt in range(MAX_RETRY + 1):
        attempts = attempt + 1
        try:
            r = requests.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=TIMEOUT_S
            )
            status = r.status_code
            r.raise_for_status()

            raw = r.json()["choices"][0]["message"]["content"]
            last_raw = raw
            out = extract_json_from_content(raw)
            out = validate_output(out)

            latency = round(time.time() - start, 3)
            meta = {
                "attempts": attempts,
                "status_code": status,
                "latency_sec": latency,
                "error": ""
            }
            return True, out, raw, meta

        except Exception as e:
            last_err = f"{type(e).__name__}:{str(e)[:300]}"
            time.sleep(1.0 * (attempt + 1))

    latency = round(time.time() - start, 3)
    meta = {
        "attempts": attempts,
        "status_code": None,
        "latency_sec": latency,
        "error": last_err[:300] if last_err else "unknown_error"
    }
    return False, None, last_raw, meta


# ===================== 构造输入 =====================
def compute_region_supply_demand(T_task_raw: pd.DataFrame, T_features_struct: pd.DataFrame) -> dict:
    df = T_features_struct.merge(T_task_raw[["task_id", "publish_time"]], on="task_id", how="left")
    g = df.groupby("region_id")
    out = {}
    for rid, sub in g:
        out[str(rid)] = {
            "task_count": int(len(sub)),
            "avg_reward": float(sub["reward"].mean()) if "reward" in sub.columns else 0.0,
            "avg_distance_km": float(sub["distance_km"].mean()) if "distance_km" in sub.columns else 0.0,
            "peak_ratio": float(sub["peak_flag"].mean()) if "peak_flag" in sub.columns else 0.0
        }
    return out


def build_dpi_payload(task_row: pd.Series, struct_row: pd.Series, region_stats: Optional[dict] = None) -> dict:
    payload = {
        "task_id": str(task_row["task_id"]),
        "task_text": str(task_row.get("task_text", "")),
        "reward": safe_float(struct_row.get("reward", np.nan)),
        "distance_km": safe_float(struct_row.get("distance_km", np.nan)),
        "time_window_sec": int(struct_row.get("time_window_sec", 0)),
        "peak_flag": int(struct_row.get("peak_flag", 0)),
        "region": str(struct_row.get("region_id", "")),
        "region_density": safe_float(struct_row.get("region_density", np.nan)),
        "worker_state": safe_float(struct_row.get("worker_state", np.nan)),
    }

    if region_stats is not None:
        rid = payload["region"]
        if rid in region_stats:
            payload["region_supply_demand"] = region_stats[rid]
    return payload


# ===================== QC 抽检 =====================
def consistency_check(sample_payloads: List[dict], baseline_rows: Optional[pd.DataFrame] = None):
    results = []
    for p in sample_payloads:
        ok1, out1, raw1, meta1 = call_deepseek_strict_json(p)
        ok2, out2, raw2, meta2 = call_deepseek_strict_json(p)

        row = {
            "task_id": p["task_id"],
            "ok1": ok1,
            "ok2": ok2,
            "attempts1": meta1["attempts"],
            "attempts2": meta2["attempts"],
            "latency1": meta1["latency_sec"],
            "latency2": meta2["latency_sec"],
        }

        if ok1 and ok2:
            keys = [
                "semantic_difficulty", "environment_complexity", "path_uncertainty",
                "rejection_risk", "sender_urgency", "alpha", "eta"
            ]
            diffs = {f"diff_{k}": abs(float(out1[k]) - float(out2[k])) for k in keys}
            row.update(diffs)
            row["mean_diff"] = float(np.mean(list(diffs.values())))
            row["same_dominant_side"] = int(
                out1["causal_cot"]["dominant_side"] == out2["causal_cot"]["dominant_side"]
            )
        else:
            row["mean_diff"] = np.nan
            row["same_dominant_side"] = np.nan

        if baseline_rows is not None:
            b = baseline_rows[baseline_rows["task_id"] == p["task_id"]]
            if len(b) == 1 and ok1:
                bp = b.iloc[0].to_dict()
                dist = float(bp.get("distance_km", 0.0))
                tw = float(bp.get("time_window_sec", 1.0))
                peak = float(bp.get("peak_flag", 0.0))
                proxy = 1 / (1 + np.exp(-(0.6 * dist + 0.8 * (1.0 / max(tw, 1)) + 0.3 * peak)))
                row["rule_difficulty_proxy"] = float(proxy)
                row["llm_semantic_difficulty"] = float(out1["semantic_difficulty"])
                row["abs_rule_llm_diff"] = abs(row["rule_difficulty_proxy"] - row["llm_semantic_difficulty"])

        results.append(row)

    return pd.DataFrame(results)


# ===================== 论文表自动生成 =====================
def build_task_feature_summary(df_row: pd.Series) -> str:
    parts = []

    reward = df_row.get("reward")
    distance_km = df_row.get("distance_km")
    time_window_sec = df_row.get("time_window_sec")
    peak_flag = df_row.get("peak_flag")
    task_text = df_row.get("task_text")

    if pd.notna(reward):
        parts.append(f"报酬{round(float(reward), 2)}")
    if pd.notna(distance_km):
        parts.append(f"距离{round(float(distance_km), 2)}km")
    if pd.notna(time_window_sec):
        parts.append(f"时窗{int(float(time_window_sec))}s")
    if pd.notna(peak_flag):
        parts.append("高峰期" if int(float(peak_flag)) == 1 else "非高峰期")
    if pd.notna(task_text):
        text = str(task_text).strip().replace("\n", " ")
        if len(text) > 24:
            text = text[:24] + "..."
        parts.append(text)

    return "，".join(parts)


def generate_case_table(base_tasks: pd.DataFrame, base_struct: pd.DataFrame):
    if not os.path.exists(OUT_LLMZ):
        return

    df_res = pd.read_csv(OUT_LLMZ)
    df_task = base_tasks.copy()
    df_struct = base_struct.copy()

    df = df_res.merge(df_task[["task_id", "task_text"]], on="task_id", how="left")
    merge_cols = ["task_id", "reward", "distance_km", "time_window_sec", "peak_flag"]
    for c in merge_cols:
        if c not in df_struct.columns:
            df_struct[c] = np.nan
    df = df.merge(df_struct[merge_cols], on="task_id", how="left")

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["eta"] = pd.to_numeric(df["eta"], errors="coerce")
    df["balance_gap"] = (df["alpha"] - df["eta"]).abs()

    sender_df = df[df["cot_dominant_side"] == "sender"].sort_values("eta", ascending=False).head(2)
    worker_df = df[df["cot_dominant_side"] == "worker"].sort_values("alpha", ascending=False).head(2)
    balanced_df = df.sort_values("balance_gap", ascending=True).head(2)

    case_df = pd.concat([sender_df, worker_df, balanced_df], axis=0).drop_duplicates(subset=["task_id"]).copy()
    case_df["任务特征概述"] = case_df.apply(build_task_feature_summary, axis=1)

    def label_type(row):
        if row["balance_gap"] < 0.05:
            return "双边平衡"
        if row["cot_dominant_side"] == "sender":
            return "发包人主导"
        return "接包人主导"

    case_df["案例类型"] = case_df.apply(label_type, axis=1)

    out_df = case_df[
        ["task_id", "案例类型", "任务特征概述", "cot_dominant_side", "cot_key_path", "alpha", "eta", "cot_weight_reason", "cot_confidence"]
    ].copy()

    out_df = out_df.rename(columns={
        "task_id": "任务ID",
        "cot_dominant_side": "dominant_side",
        "cot_key_path": "key_causal_path",
        "cot_weight_reason": "解释说明",
        "cot_confidence": "置信度",
    })
    out_df.to_csv(OUT_CASE_TABLE, index=False, encoding="utf-8-sig")


def generate_summary_table():
    if not os.path.exists(OUT_LLMZ):
        return

    df = pd.read_csv(OUT_LLMZ)
    if len(df) == 0:
        return

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["eta"] = pd.to_numeric(df["eta"], errors="coerce")
    df["cot_confidence"] = pd.to_numeric(df["cot_confidence"], errors="coerce")

    sender_ratio = (df["cot_dominant_side"] == "sender").mean()
    worker_ratio = (df["cot_dominant_side"] == "worker").mean()
    eta_gt_alpha_ratio = (df["eta"] > df["alpha"]).mean()
    alpha_gt_eta_ratio = (df["alpha"] > df["eta"]).mean()
    approx_equal_ratio = ((df["alpha"] - df["eta"]).abs() < 0.05).mean()
    consistency = (
        ((df["cot_dominant_side"] == "sender") & (df["eta"] > df["alpha"])) |
        ((df["cot_dominant_side"] == "worker") & (df["alpha"] > df["eta"]))
    ).mean()

    summary_df = pd.DataFrame([{
        "模型": DEEPSEEK_MODEL,
        "sender主导占比": round(float(sender_ratio), 4),
        "worker主导占比": round(float(worker_ratio), 4),
        "eta>alpha占比": round(float(eta_gt_alpha_ratio), 4),
        "alpha>eta占比": round(float(alpha_gt_eta_ratio), 4),
        "alpha≈eta占比": round(float(approx_equal_ratio), 4),
        "主导侧与权重一致性": round(float(consistency), 4),
        "平均置信度": round(float(df["cot_confidence"].mean()), 4),
        "样本量": int(len(df)),
    }])
    summary_df.to_csv(OUT_SUMMARY_TABLE, index=False, encoding="utf-8-sig")


# ===================== 落盘与报告 =====================
def _flush(old_llmz, old_log, llm_rows, log_rows):
    if llm_rows:
        df = pd.DataFrame(llm_rows)
        append_csv(OUT_LLMZ, df)

        old_llmz.drop(old_llmz.index, inplace=True)
        for c in df.columns:
            old_llmz[c] = df[c]

    if log_rows:
        df = pd.DataFrame(log_rows)
        append_csv(OUT_LOG, df)

        old_log.drop(old_log.index, inplace=True)
        for c in df.columns:
            old_log[c] = df[c]


def build_qc_report():
    if not os.path.exists(OUT_LOG):
        return {"error": "no_log"}

    log = pd.read_csv(OUT_LOG)
    total = len(log)
    ok = int(log["ok"].sum()) if "ok" in log.columns else 0
    success_rate = float(ok / total) if total else 0.0

    retry = log["attempts"].fillna(1).astype(int)
    retry_rate = float((retry > 1).mean()) if total else 0.0
    avg_attempts = float(retry.mean()) if total else 0.0
    avg_latency = float(log["latency_sec"].mean()) if "latency_sec" in log.columns and total else 0.0

    report = {
        "timestamp_utc": now_iso(),
        "model": DEEPSEEK_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
        "total_requests": total,
        "success_requests": ok,
        "format_success_rate": success_rate,
        "retry_rate": retry_rate,
        "avg_attempts": avg_attempts,
        "avg_latency_sec": avg_latency,
        "llmz_file": OUT_LLMZ,
        "log_file": OUT_LOG,
        "case_table": OUT_CASE_TABLE,
        "summary_table": OUT_SUMMARY_TABLE,
    }

    if os.path.exists(OUT_LLMZ):
        z = pd.read_csv(OUT_LLMZ)
        keys = [
            "semantic_difficulty", "environment_complexity", "path_uncertainty",
            "rejection_risk", "sender_urgency", "alpha", "eta", "cot_confidence"
        ]
        range_stats = {}
        for k in keys:
            if k in z.columns:
                s = pd.to_numeric(z[k], errors="coerce")
                range_stats[k] = {
                    "min": float(np.nanmin(s.values)),
                    "max": float(np.nanmax(s.values)),
                    "mean": float(np.nanmean(s.values)),
                    "nan_rate": float(np.mean(~np.isfinite(s.values)))
                }
        report["field_range_stats"] = range_stats

    return report


def do_qc_sampling(T_task_raw, T_features_struct, region_stats, baseline_rows):
    if not os.path.exists(OUT_LLMZ):
        return pd.DataFrame()

    llmz = pd.read_csv(OUT_LLMZ)
    llmz["task_id"] = llmz["task_id"].astype(str)
    candidates = llmz["task_id"].tolist()
    if len(candidates) == 0:
        return pd.DataFrame()

    n = min(QC_SAMPLE_N, len(candidates))
    sample_ids = random.sample(candidates, k=n)

    payloads = []
    for tid in sample_ids:
        task_row = T_task_raw[T_task_raw["task_id"].astype(str) == tid].iloc[0]
        struct_row = T_features_struct[T_features_struct["task_id"].astype(str) == tid].iloc[0]
        payloads.append(build_dpi_payload(task_row, struct_row, region_stats=region_stats))

    df = consistency_check(payloads, baseline_rows=baseline_rows)
    if len(df):
        df["stable_flag"] = (df["mean_diff"] <= 0.10).astype(int)
    return df


# ===================== 主流程 =====================
def main():
    ensure_dir(OUT_DIR)

    T_task_raw = pd.read_csv(TASK_FILE)
    T_features_struct = pd.read_csv(STRUCT_FILE)

    T_task_raw["task_id"] = T_task_raw["task_id"].astype(str)
    T_features_struct["task_id"] = T_features_struct["task_id"].astype(str)

    region_stats = compute_region_supply_demand(T_task_raw, T_features_struct)

    old_llmz = load_existing_llmz()
    done_ids = set(old_llmz["task_id"].astype(str).tolist()) if len(old_llmz) else set()

    old_log = load_existing_log()

    print(f"[Resume] already done tasks: {len(done_ids)}")

    tasks = T_task_raw["task_id"].astype(str).tolist()
    todo = [tid for tid in tasks if tid not in done_ids]
    print(f"[Todo] remaining tasks: {len(todo)}")

    llm_rows = []
    log_rows = []

    baseline_rows = T_features_struct.copy()
    baseline_rows["task_id"] = baseline_rows["task_id"].astype(str)

    processed_since_flush = 0

    for tid in todo:
        task_row = T_task_raw[T_task_raw["task_id"] == tid].iloc[0]
        struct_row = T_features_struct[T_features_struct["task_id"] == tid].iloc[0]

        payload = build_dpi_payload(task_row, struct_row, region_stats=region_stats)
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_hash = sha256_text(payload_str)

        tstamp = now_iso()
        ok, out, raw, meta = call_deepseek_strict_json(payload)

        if ok:
            llm_rows.append({
                "task_id": tid,
                "semantic_difficulty": float(out["semantic_difficulty"]),
                "environment_complexity": float(out["environment_complexity"]),
                "path_uncertainty": float(out["path_uncertainty"]),
                "rejection_risk": float(out["rejection_risk"]),
                "sender_urgency": float(out["sender_urgency"]),
                "alpha": float(out["alpha"]),
                "eta": float(out["eta"]),
                "rationale": str(out["rationale"]),
                "cot_dominant_side": str(out["causal_cot"]["dominant_side"]),
                "cot_key_path": str(out["causal_cot"]["key_causal_path"]),
                "cot_secondary_path": str(out["causal_cot"]["secondary_path"]),
                "cot_weight_reason": str(out["causal_cot"]["weight_reason"]),
                "cot_decision_rule": str(out["causal_cot"]["decision_rule"]),
                "cot_confidence": float(out["causal_cot"]["cot_score_confidence"]),
                "model": DEEPSEEK_MODEL,
                "base_url": DEEPSEEK_BASE_URL
            })

        log_rows.append({
            "task_id": tid,
            "timestamp_utc": tstamp,
            "model": DEEPSEEK_MODEL,
            "base_url": DEEPSEEK_BASE_URL,
            "ok": int(ok),
            "attempts": meta["attempts"],
            "latency_sec": meta["latency_sec"],
            "error": meta["error"],
            "payload_hash": payload_hash,
            "payload": payload_str,
            "output_raw": raw if raw else "",
        })

        processed_since_flush += 1
        if processed_since_flush % 50 == 0:
            _flush(old_llmz, old_log, llm_rows, log_rows)
            llm_rows.clear()
            log_rows.clear()
            print(f"[Progress] flushed {processed_since_flush} items ...")

    _flush(old_llmz, old_log, llm_rows, log_rows)

    qc_report = build_qc_report()
    with open(OUT_QC, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, ensure_ascii=False, indent=2)

    sample_df = do_qc_sampling(T_task_raw, T_features_struct, region_stats, baseline_rows)
    sample_df.to_csv(OUT_QC_SAMPLE, index=False, encoding="utf-8-sig")

    generate_case_table(T_task_raw, T_features_struct)
    generate_summary_table()

    print("✅ Step2 finished.")
    print("LLMZ:", OUT_LLMZ)
    print("LOG :", OUT_LOG)
    print("QC  :", OUT_QC)
    print("QC_sample:", OUT_QC_SAMPLE)
    print("Case table:", OUT_CASE_TABLE)
    print("Summary table:", OUT_SUMMARY_TABLE)


if __name__ == "__main__":
    main()