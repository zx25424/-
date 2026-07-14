# -*- coding: utf-8 -*-
import os
import json
import time
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI

from utils_io import ensure_dir, read_csv_auto, write_json, sha256_text
from prompt_templates import SYSTEM_PROMPT, build_user_prompt

DATA_DIR = os.getenv("DATA_DIR", "sim_experiment_data")
OUT_DIR = os.getenv("OUT_DIR", DATA_DIR)

STRUCT_FILE = os.path.join(DATA_DIR, "T_features_struct.csv")
OUT_FILE = os.path.join(OUT_DIR, "T_features_llmZ.csv")
LOG_FILE = os.path.join(OUT_DIR, "T_api_log.csv")
QC_FILE = os.path.join(OUT_DIR, "QC_report.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

DPI_TEMPERATURE = float(os.getenv("DPI_TEMPERATURE", "0.2"))
DPI_MAX_RETRY = int(os.getenv("DPI_MAX_RETRY", "2"))
DPI_TIMEOUT = int(os.getenv("DPI_TIMEOUT", "60"))
RESUME = int(os.getenv("RESUME", "1"))


REQUIRED_FIELDS = [
    "semantic_difficulty",
    "environment_complexity",
    "path_uncertainty",
    "rejection_risk",
    "sender_urgency",
    "alpha",
    "eta",
    "dominant_side",
    "primary_causal_path",
    "secondary_causal_path",
    "weight_reason",
    "decision_rule",
    "confidence",
    "rationale",
]


def get_client():
    if not OPENAI_API_KEY:
        raise ValueError("缺少 OPENAI_API_KEY 环境变量")
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=DPI_TIMEOUT,
    )


def parse_json_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3].strip()
    return json.loads(text)


def validate_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    for k in REQUIRED_FIELDS:
        if k not in obj:
            raise ValueError(f"缺少字段: {k}")

    for k in [
        "semantic_difficulty", "environment_complexity", "path_uncertainty",
        "rejection_risk", "sender_urgency", "alpha", "eta", "confidence"
    ]:
        v = float(obj[k])
        if v < 0 or v > 1:
            raise ValueError(f"{k} 超出[0,1]: {v}")
        obj[k] = v

    if obj["dominant_side"] not in ["sender", "worker"]:
        raise ValueError("dominant_side 只能是 sender 或 worker")

    for k in [
        "primary_causal_path", "secondary_causal_path",
        "weight_reason", "decision_rule", "rationale"
    ]:
        obj[k] = str(obj[k]).strip()

    return obj


def call_gpt4o(client: OpenAI, row: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(row)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=DPI_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    obj = parse_json_text(content)
    obj = validate_obj(obj)
    return obj, content, user_prompt


def load_existing_ids() -> set:
    if RESUME == 1 and os.path.exists(OUT_FILE):
        df = read_csv_auto(OUT_FILE)
        if "task_id" in df.columns and "worker_id" in df.columns:
            return set(zip(df["task_id"].astype(str), df["worker_id"].astype(str)))
    return set()


def main():
    ensure_dir(OUT_DIR)

    df = read_csv_auto(STRUCT_FILE)
    df["task_id"] = df["task_id"].astype(str)
    df["worker_id"] = df["worker_id"].astype(str)

    existing_ids = load_existing_ids()
    client = get_client()

    logs: List[Dict[str, Any]] = []
    rows_out: List[Dict[str, Any]] = []

    if existing_ids:
        print(f"[Resume] 已存在记录数: {len(existing_ids)}")

    total = len(df)
    start_time_all = time.time()

    for i, row in df.iterrows():
        key = (str(row["task_id"]), str(row["worker_id"]))
        if key in existing_ids:
            continue

        row_dict = row.to_dict()
        ok = False
        last_err = ""
        raw_content = ""
        prompt_hash = ""
        latency = None

        for attempt in range(1, DPI_MAX_RETRY + 2):
            t0 = time.time()
            try:
                obj, raw_content, user_prompt = call_gpt4o(client, row_dict)
                latency = time.time() - t0
                prompt_hash = sha256_text(user_prompt)

                out_row = dict(row_dict)
                out_row.update(obj)
                rows_out.append(out_row)

                logs.append({
                    "task_id": row_dict["task_id"],
                    "worker_id": row_dict["worker_id"],
                    "attempt": attempt,
                    "status": "success",
                    "latency_sec": latency,
                    "prompt_hash": prompt_hash,
                    "raw_output": raw_content,
                    "error_message": "",
                    "model": OPENAI_MODEL,
                })
                ok = True
                break
            except Exception as e:
                latency = time.time() - t0
                last_err = str(e)
                logs.append({
                    "task_id": row_dict["task_id"],
                    "worker_id": row_dict["worker_id"],
                    "attempt": attempt,
                    "status": "error",
                    "latency_sec": latency,
                    "prompt_hash": prompt_hash,
                    "raw_output": raw_content,
                    "error_message": last_err,
                    "model": OPENAI_MODEL,
                })
                time.sleep(1.0)

        if not ok:
            fallback = dict(row_dict)
            fallback.update({
                "semantic_difficulty": 0.5,
                "environment_complexity": 0.5,
                "path_uncertainty": min(float(row_dict.get("route_diff_ratio", 0.0)), 1.0),
                "rejection_risk": min(float(row_dict.get("task_difficulty", 0.0)), 1.0),
                "sender_urgency": 0.6 if int(row_dict.get("is_prebook", 0)) == 0 else 0.4,
                "alpha": 0.5,
                "eta": 0.5,
                "dominant_side": "worker",
                "primary_causal_path": "任务难度上升导致执行压力增加",
                "secondary_causal_path": "路径不确定性增加导致接单意愿下降",
                "weight_reason": "基础回退权重",
                "decision_rule": "在API失败时采用默认均衡策略",
                "confidence": 0.3,
                "rationale": "API失败后的回退结果",
            })
            rows_out.append(fallback)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time_all
            done = i + 1
            avg = elapsed / max(done, 1)
            remain = avg * (total - done) / 60.0
            print(f"[Progress] 已处理 {done}/{total} | 平均 {avg:.2f} 秒/条 | 预计剩余 {remain:.1f} 分钟")

    old_df = pd.DataFrame()
    if os.path.exists(OUT_FILE):
        try:
            old_df = read_csv_auto(OUT_FILE)
        except Exception:
            old_df = pd.DataFrame()

    new_df = pd.DataFrame(rows_out)
    final_df = pd.concat([old_df, new_df], axis=0, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["task_id", "worker_id"], keep="last")

    log_df = pd.DataFrame(logs)
    if os.path.exists(LOG_FILE):
        try:
            old_log = read_csv_auto(LOG_FILE)
            log_df = pd.concat([old_log, log_df], axis=0, ignore_index=True)
        except Exception:
            pass

    final_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    log_df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

    qc = {
        "total_rows": int(len(df)),
        "generated_rows": int(len(final_df)),
        "success_logs": int((log_df["status"] == "success").sum()) if len(log_df) else 0,
        "error_logs": int((log_df["status"] == "error").sum()) if len(log_df) else 0,
        "success_rate_by_logs": float((log_df["status"] == "success").mean()) if len(log_df) else 0.0,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
    }
    write_json(QC_FILE, qc)

    print(f"[OK] Wrote: {OUT_FILE}")
    print(f"[OK] Wrote: {LOG_FILE}")
    print(f"[OK] Wrote: {QC_FILE}")
    print("[DONE] step2 finished.")


if __name__ == "__main__":
    main()