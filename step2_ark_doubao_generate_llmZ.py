# step2_ark_doubao_generate_llmZ.py
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import hashlib
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sim_experiment_data")
OUT_DIR = DATA_DIR

TASK_FILE = os.path.join(DATA_DIR, "T_task_raw.csv")
STRUCT_FILE = os.path.join(DATA_DIR, "T_features_struct.csv")

OUT_LLMZ = os.path.join(OUT_DIR, "T_features_llmZ.csv")
OUT_LOG = os.path.join(OUT_DIR, "T_dpi_log.csv")
OUT_QC = os.path.join(OUT_DIR, "QC_report.json")

# =========================
# 环境变量配置
# =========================
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "")
ARK_MODEL = os.getenv("ARK_MODEL", "doubao-seed-1-8-251228")


TEMPERATURE = float(os.getenv("DPI_TEMPERATURE", "0.2"))
MAX_RETRY = int(os.getenv("DPI_MAX_RETRY", "1"))
QC_SAMPLE_N = int(os.getenv("QC_SAMPLE_N", "200"))
SEED = int(os.getenv("SEED", "0"))
DEBUG_N = int(os.getenv("DEBUG_N", "0"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "20"))

random.seed(SEED)
np.random.seed(SEED)

SYSTEM_PROMPT = (
    "你是众包物流平台实验中的因果特征生成器。"
    "你必须只输出严格JSON，不允许额外解释。"
    "JSON必须包含字段："
    "semantic_difficulty, environment_complexity, path_uncertainty, "
    "rejection_risk, sender_urgency, alpha, eta, rationale。"
    "其中前7个数值字段必须在0到1之间，rationale是简短中文说明。"
)

LLMZ_COLS = [
    "task_id",
    "semantic_difficulty",
    "environment_complexity",
    "path_uncertainty",
    "rejection_risk",
    "sender_urgency",
    "alpha",
    "eta",
    "rationale",
    "model"
]

LOG_COLS = [
    "task_id",
    "timestamp_utc",
    "model",
    "ok",
    "payload_hash",
    "payload",
    "output_raw",
    "error"
]

thread_local = threading.local()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_json_lenient(raw: str):
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1:
        raise ValueError("No JSON object found")
    return json.loads(raw[l:r + 1])


def build_payload(task_row, struct_row):
    return {
        "task_id": str(task_row["task_id"]),
        "task_text": str(task_row["task_text"]),
        "reward": float(struct_row["reward"]),
        "distance_km": float(struct_row["distance_km"]),
        "time_window_sec": float(struct_row["time_window_sec"]),
        "peak_flag": int(struct_row["peak_flag"]),
        "region_id": str(struct_row["region_id"]),
        "task_count": int(struct_row["task_count"]),
        "path_error": float(struct_row.get("path_error", 0.0)),
        "task_difficulty_struct": float(struct_row.get("task_difficulty_struct", 0.5)),
        "region_density": float(struct_row.get("region_density", 0.0))
    }


def save_partial(rows, logs):
    pd.DataFrame(rows, columns=LLMZ_COLS).to_csv(
        OUT_LLMZ, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(logs, columns=LOG_COLS).to_csv(
        OUT_LOG, index=False, encoding="utf-8-sig"
    )


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            timeout=60
        )
    return thread_local.client


def process_one(task_dict, struct_map):
    tid = str(task_dict["task_id"])
    srow = struct_map.get(tid)

    if srow is None:
        return {
            "tid": tid,
            "ok": 0,
            "row_result": None,
            "log_result": {
                "task_id": tid,
                "timestamp_utc": now_iso(),
                "model": ARK_MODEL,
                "ok": 0,
                "payload_hash": "",
                "payload": "",
                "output_raw": "",
                "error": "struct_not_found"
            }
        }

    payload = build_payload(task_dict, srow)
    payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    ok = False
    out_json = None
    raw = ""
    err = ""

    for attempt in range(MAX_RETRY + 1):
        try:
            # 增加轻微随机抖动，避免多个线程同一时刻撞接口
            time.sleep(random.uniform(0.1, 0.5))

            client = get_client()
            resp = client.chat.completions.create(
                model=ARK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                temperature=TEMPERATURE
            )

            raw = resp.choices[0].message.content.strip()
            out_json = parse_json_lenient(raw)

            required = [
                "semantic_difficulty",
                "environment_complexity",
                "path_uncertainty",
                "rejection_risk",
                "sender_urgency",
                "alpha",
                "eta",
                "rationale"
            ]

            for k in required:
                if k not in out_json:
                    raise ValueError(f"missing_field:{k}")

            for k in required[:-1]:
                v = float(out_json[k])
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"out_of_range:{k}:{v}")

            ok = True
            break

        except Exception as e:
            err = str(e)
            if attempt < MAX_RETRY:
                # 轻度指数退避
                sleep_sec = 2.0 * (attempt + 1) + random.uniform(0.2, 0.8)
                time.sleep(sleep_sec)

    row_result = None
    if ok:
        row_result = {
            "task_id": tid,
            "semantic_difficulty": float(out_json["semantic_difficulty"]),
            "environment_complexity": float(out_json["environment_complexity"]),
            "path_uncertainty": float(out_json["path_uncertainty"]),
            "rejection_risk": float(out_json["rejection_risk"]),
            "sender_urgency": float(out_json["sender_urgency"]),
            "alpha": float(out_json["alpha"]),
            "eta": float(out_json["eta"]),
            "rationale": str(out_json["rationale"]),
            "model": ARK_MODEL
        }

    log_result = {
        "task_id": tid,
        "timestamp_utc": now_iso(),
        "model": ARK_MODEL,
        "ok": int(ok),
        "payload_hash": sha256_text(payload_str),
        "payload": payload_str,
        "output_raw": raw,
        "error": err
    }

    return {
        "tid": tid,
        "ok": int(ok),
        "row_result": row_result,
        "log_result": log_result
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("当前工作目录:", os.getcwd())
    print("脚本目录:", BASE_DIR)
    print("数据目录:", DATA_DIR)
    print("输出目录:", OUT_DIR)
    print("TASK_FILE:", TASK_FILE)
    print("STRUCT_FILE:", STRUCT_FILE)
    print("TASK_FILE存在吗:", os.path.exists(TASK_FILE))
    print("STRUCT_FILE存在吗:", os.path.exists(STRUCT_FILE))
    print("OUT_LLMZ:", OUT_LLMZ)
    print("OUT_LOG:", OUT_LOG)
    print("OUT_QC:", OUT_QC)

    if not ARK_API_KEY or not ARK_BASE_URL:
        raise RuntimeError("请先设置 ARK_API_KEY 和 ARK_BASE_URL")

    if not os.path.exists(TASK_FILE):
        raise FileNotFoundError(f"找不到任务文件: {TASK_FILE}")

    if not os.path.exists(STRUCT_FILE):
        raise FileNotFoundError(f"找不到结构特征文件: {STRUCT_FILE}")

    task = pd.read_csv(TASK_FILE)
    struct = pd.read_csv(STRUCT_FILE)

    task["task_id"] = task["task_id"].astype(str).str.strip()
    struct["task_id"] = struct["task_id"].astype(str).str.strip()

    if DEBUG_N > 0:
        task = task.head(DEBUG_N).copy()
        print(f"调试模式：只处理前 {len(task)} 条任务")

    print("任务数:", len(task))
    print("结构表数:", len(struct))
    print("MAX_RETRY:", MAX_RETRY)
    print("TEMPERATURE:", TEMPERATURE)
    print("ARK_MODEL:", ARK_MODEL)
    print("MAX_WORKERS:", MAX_WORKERS)
    print("SAVE_EVERY:", SAVE_EVERY)

    # 检查 task_id 匹配情况
    task_ids = set(task["task_id"])
    struct_ids = set(struct["task_id"])
    matched_ids = task_ids & struct_ids
    missing_in_struct = task_ids - struct_ids

    print("task总数:", len(task_ids))
    print("struct中task_id总数:", len(struct_ids))
    print("可匹配task_id数:", len(matched_ids))
    print("在struct中找不到的task_id数:", len(missing_in_struct))

    struct_map = struct.set_index("task_id").to_dict(orient="index")

    rows = []
    logs = []
    done_ids = set()

    if os.path.exists(OUT_LLMZ):
        try:
            old_llmz = pd.read_csv(OUT_LLMZ)
            if "task_id" in old_llmz.columns:
                old_llmz["task_id"] = old_llmz["task_id"].astype(str).str.strip()
                rows = old_llmz.to_dict(orient="records")
                print(f"检测到历史 LLMZ 结果: {len(rows)} 条")
        except Exception as e:
            print("读取历史 T_features_llmZ.csv 失败:", e)

    if os.path.exists(OUT_LOG):
        try:
            old_log = pd.read_csv(OUT_LOG)
            if "task_id" in old_log.columns:
                old_log["task_id"] = old_log["task_id"].astype(str).str.strip()
                logs = old_log.to_dict(orient="records")
                done_ids = set(old_log["task_id"].tolist())
                print(f"检测到历史日志，已完成任务数: {len(done_ids)}")
        except Exception as e:
            print("读取历史 T_dpi_log.csv 失败:", e)

    save_partial(rows, logs)
    print("已初始化输出文件:", OUT_LLMZ)
    print("已初始化日志文件:", OUT_LOG)

    task_records = task.to_dict(orient="records")
    pending_tasks = [x for x in task_records if str(x["task_id"]).strip() not in done_ids]

    print(f"待处理任务数: {len(pending_tasks)}")

    if len(pending_tasks) == 0:
        llmz = pd.DataFrame(rows, columns=LLMZ_COLS)
        log = pd.DataFrame(logs, columns=LOG_COLS)
        report = {
            "total_requests": int(len(log)),
            "success_requests": int(log["ok"].sum()) if len(log) else 0,
            "success_rate": float(log["ok"].mean()) if len(log) else 0.0,
            "sample_n": min(QC_SAMPLE_N, len(llmz)),
            "model": ARK_MODEL
        }
        with open(OUT_QC, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("✅ Step2 finished.")
        return

    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, task_dict, struct_map): str(task_dict["task_id"]).strip()
            for task_dict in pending_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Doubao concurrent Z"):
            tid = futures[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "tid": tid,
                    "ok": 0,
                    "row_result": None,
                    "log_result": {
                        "task_id": tid,
                        "timestamp_utc": now_iso(),
                        "model": ARK_MODEL,
                        "ok": 0,
                        "payload_hash": "",
                        "payload": "",
                        "output_raw": "",
                        "error": f"future_exception:{str(e)}"
                    }
                }

            if result["row_result"] is not None:
                rows.append(result["row_result"])

            logs.append(result["log_result"])
            completed_count += 1

            if completed_count % SAVE_EVERY == 0 or completed_count == len(pending_tasks):
                save_partial(rows, logs)

                elapsed = time.time() - start_time
                avg = elapsed / completed_count if completed_count > 0 else 0.0
                remain = avg * (len(pending_tasks) - completed_count)

                current_success = sum(1 for x in logs if int(x.get("ok", 0)) == 1)
                current_fail = sum(1 for x in logs if int(x.get("ok", 0)) == 0)
                struct_missing = sum(1 for x in logs if str(x.get("error", "")) == "struct_not_found")
                conn_error = sum(1 for x in logs if "Connection error" in str(x.get("error", "")))

                print(
                    f"已完成 {completed_count}/{len(pending_tasks)} | "
                    f"累计成功 {current_success} | "
                    f"累计失败 {current_fail} | "
                    f"struct缺失 {struct_missing} | "
                    f"连接错误 {conn_error} | "
                    f"平均 {avg:.2f} 秒/条 | "
                    f"预计剩余 {remain / 60:.1f} 分钟"
                )

    llmz = pd.DataFrame(rows, columns=LLMZ_COLS)
    log = pd.DataFrame(logs, columns=LOG_COLS)

    llmz.to_csv(OUT_LLMZ, index=False, encoding="utf-8-sig")
    log.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

    report = {
        "total_requests": int(len(log)),
        "success_requests": int(log["ok"].sum()) if len(log) else 0,
        "success_rate": float(log["ok"].mean()) if len(log) else 0.0,
        "sample_n": min(QC_SAMPLE_N, len(llmz)),
        "model": ARK_MODEL
    }

    with open(OUT_QC, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start_time
    print("✅ Step2 finished.")
    print("T_features_llmZ:", OUT_LLMZ)
    print("T_dpi_log:", OUT_LOG)
    print("QC_report:", OUT_QC)
    print(f"总耗时: {total_elapsed / 60:.2f} 分钟")


if __name__ == "__main__":
    main()