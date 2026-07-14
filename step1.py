# step1_data_prepare_and_split.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ 配置 ============
DATA_DIR = "C:\\Users\\lenovo\\Desktop\\小论文1\\step1\\pythonProject\\数据集"   # 改成你的 6 张表所在目录
OUT_DIR = "C:\\Users\\lenovo\\Desktop\\小论文1\\step1\\pythonProject\\输出"          # 输出目录

TASK_FILE = os.path.join(DATA_DIR, "T_task_raw.csv")
INTER_FILE = os.path.join(DATA_DIR, "T_interaction.csv")
STRUCT_FILE = os.path.join(DATA_DIR, "T_features_struct.csv")
OUTCOME_FILE = os.path.join(DATA_DIR, "T_outcomes.csv")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "splits"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "stats"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)


# ============ 1) 读取数据 ============
T_task_raw = pd.read_csv(TASK_FILE)
T_interaction = pd.read_csv(INTER_FILE)
T_features_struct = pd.read_csv(STRUCT_FILE)
T_outcomes = pd.read_csv(OUTCOME_FILE)

# 统一 task_id / worker_id 类型（避免 merge/groupby 出现“1”和“001”的问题）
for df in [T_task_raw, T_interaction, T_features_struct, T_outcomes]:
    if "task_id" in df.columns:
        df["task_id"] = df["task_id"].astype(str)

if "worker_id" in T_interaction.columns:
    T_interaction["worker_id"] = T_interaction["worker_id"].astype(str)
if "worker_id" in T_outcomes.columns:
    T_outcomes["worker_id"] = T_outcomes["worker_id"].astype(str)

# publish_time 解析（T_task_raw 一定要有）
T_task_raw["publish_time"] = pd.to_datetime(T_task_raw["publish_time"], errors="coerce")
if "publish_time" in T_interaction.columns:
    T_interaction["publish_time"] = pd.to_datetime(T_interaction["publish_time"], errors="coerce")

# 去掉 publish_time 缺失的任务（无法做时间切分）
task_times = (
    T_task_raw[["task_id", "publish_time"]]
    .dropna(subset=["publish_time"])
    .sort_values("publish_time")
    .reset_index(drop=True)
)

assert len(task_times) > 0, "T_task_raw.publish_time 全部为空，无法切分。"

# ============ 2) 按 publish_time 时间切分：70/10/20 ============
n = len(task_times)
train_cut = int(n * 0.7)
val_cut = int(n * 0.8)

train_tasks = task_times.iloc[:train_cut]["task_id"].tolist()
val_tasks   = task_times.iloc[train_cut:val_cut]["task_id"].tolist()
test_tasks  = task_times.iloc[val_cut:]["task_id"].tolist()

# 保存 split 索引文件
def save_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")

save_list(os.path.join(OUT_DIR, "splits", "train_tasks.txt"), train_tasks)
save_list(os.path.join(OUT_DIR, "splits", "val_tasks.txt"), val_tasks)
save_list(os.path.join(OUT_DIR, "splits", "test_tasks.txt"), test_tasks)

print(f"[Split] total={n}, train={len(train_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}")


# ============ 3) 构造 “每个 worker 在时刻 t 的候选任务集” ============
# 你提到两种方式：
# A) 从 T_interaction 按 (worker_id, publish_time) 聚合候选任务列表
# B) 或按任务发布时刻对 worker 聚合（本质同A，只是你可以做时间粒度聚合）

# --- 3.1 精确到 publish_time（原始粒度）聚合 ---
# 输出：每行 (worker_id, publish_time) -> 候选 task_id 列表、候选数
cand_by_worker_time = (
    T_interaction
    .dropna(subset=["worker_id", "publish_time", "task_id"])
    .groupby(["worker_id", "publish_time"])["task_id"]
    .agg(task_list=lambda s: list(pd.unique(s.astype(str))),
         task_count="nunique")
    .reset_index()
)
cand_by_worker_time.to_csv(os.path.join(OUT_DIR, "stats", "cand_by_worker_publish_time.csv"),
                           index=False, encoding="utf-8-sig")

print("[CandSet] cand_by_worker_publish_time rows:", len(cand_by_worker_time))

# --- 3.2 可选：按时间桶聚合（例如按分钟/5分钟/小时） ---
# 适用于你仿真时刻 t 是离散时间步的情况
TIME_BUCKET = os.getenv("TIME_BUCKET", "5min")  # 可改: "1min", "5min", "15min", "1H"
tmp = T_interaction.dropna(subset=["worker_id","publish_time","task_id"]).copy()
tmp["time_bucket"] = tmp["publish_time"].dt.floor(TIME_BUCKET)

cand_by_worker_bucket = (
    tmp.groupby(["worker_id", "time_bucket"])["task_id"]
       .agg(task_list=lambda s: list(pd.unique(s.astype(str))),
            task_count="nunique")
       .reset_index()
)
cand_by_worker_bucket.to_csv(os.path.join(OUT_DIR, "stats", f"cand_by_worker_{TIME_BUCKET}.csv"),
                             index=False, encoding="utf-8-sig")

print("[CandSet] cand_by_worker_bucket rows:", len(cand_by_worker_bucket))


# ============ 4) 候选集统计（每任务候选数、每 worker 候选数） ============
# --- 4.1 每任务候选数（候选集大小分布） ---
cand_per_task = (
    T_interaction.dropna(subset=["task_id","worker_id"])
    .groupby("task_id")["worker_id"]
    .nunique()
    .rename("num_candidates")
    .reset_index()
)
cand_per_task.to_csv(os.path.join(OUT_DIR, "stats", "cand_per_task.csv"),
                     index=False, encoding="utf-8-sig")

# --- 4.2 每 worker 候选数：两种常用口径 ---
# (a) 被曝光到多少任务（exposure count by task unique）
cand_tasks_per_worker = (
    T_interaction.dropna(subset=["worker_id","task_id"])
    .groupby("worker_id")["task_id"]
    .nunique()
    .rename("num_exposed_tasks")
    .reset_index()
)
cand_tasks_per_worker.to_csv(os.path.join(OUT_DIR, "stats", "cand_tasks_per_worker.csv"),
                             index=False, encoding="utf-8-sig")

# (b) 被曝光记录条数（如果你有 exposed 字段，可用 sum(exposed)）
if "exposed" in T_interaction.columns:
    exposed_logs_per_worker = (
        T_interaction.dropna(subset=["worker_id"])
        .groupby("worker_id")["exposed"]
        .sum()
        .rename("num_exposed_logs")
        .reset_index()
    )
    exposed_logs_per_worker.to_csv(os.path.join(OUT_DIR, "stats", "exposed_logs_per_worker.csv"),
                                   index=False, encoding="utf-8-sig")
else:
    exposed_logs_per_worker = None


# ============ 5) 分布图（默认 matplotlib 颜色，不指定颜色） ============
# --- 5.1 每任务候选数分布（直方图） ---
plt.figure()
plt.hist(cand_per_task["num_candidates"].values, bins=30)
plt.xlabel("Number of candidates per task")
plt.ylabel("Count of tasks")
plt.title("Candidate set size distribution (per task)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figs", "hist_num_candidates_per_task.png"), dpi=200)
plt.close()

# --- 5.2 每 worker 曝光任务数分布 ---
plt.figure()
plt.hist(cand_tasks_per_worker["num_exposed_tasks"].values, bins=30)
plt.xlabel("Number of exposed tasks per worker")
plt.ylabel("Count of workers")
plt.title("Exposure distribution (unique tasks per worker)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figs", "hist_num_exposed_tasks_per_worker.png"), dpi=200)
plt.close()

# --- 5.3 每 worker-时刻候选任务数分布 ---
plt.figure()
plt.hist(cand_by_worker_time["task_count"].values, bins=30)
plt.xlabel("Candidate tasks per (worker, publish_time)")
plt.ylabel("Count of (worker,time) pairs")
plt.title("Candidate tasks distribution per (worker, time)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figs", "hist_candidate_tasks_per_worker_time.png"), dpi=200)
plt.close()

# --- 5.4 每 worker-时间桶候选任务数分布（可选） ---
plt.figure()
plt.hist(cand_by_worker_bucket["task_count"].values, bins=30)
plt.xlabel(f"Candidate tasks per (worker, {TIME_BUCKET})")
plt.ylabel("Count of (worker,time_bucket) pairs")
plt.title(f"Candidate tasks distribution per (worker, {TIME_BUCKET})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figs", f"hist_candidate_tasks_per_worker_{TIME_BUCKET}.png"), dpi=200)
plt.close()

print("✅ Step1 done.")
print("Outputs:", OUT_DIR)
