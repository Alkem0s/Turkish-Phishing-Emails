import glob
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns
import pandas as pd
import numpy as np

# ============================
# CONFIG
# ============================

RESULTS_DIR = "./results"
OUTPUT_DIR = "./results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_DPI = 300
STYLE = "whitegrid"

sns.set_theme(style=STYLE)

# ============================
# LOAD RESULTS
# ============================

results_files = glob.glob(os.path.join(RESULTS_DIR, "results*.json"))
if not results_files:
    raise FileNotFoundError("No JSON file beginning with 'results' found in ./results/")

RESULTS_PATH = results_files[0]
print(f"Loading results from: {RESULTS_PATH}")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    raw_results = json.load(f)

# Flatten metrics
records = []
for approach, data in raw_results.items():
    metrics = data.get("metrics", data)
    records.append({
        "approach": approach.replace("_", " ").title(),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "auc_roc": metrics["auc_roc"],
        "confusion_matrix": metrics["confusion_matrix"]
    })

df = pd.DataFrame(records)

# ============================
# 1️ METRIC BAR PLOT
# ============================

metric_cols = ["accuracy", "f1_score", "auc_roc"]
df_melt = df.melt(
    id_vars="approach",
    value_vars=metric_cols,
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=df_melt,
    x="approach",
    y="score",
    hue="metric"
)

plt.ylim(0.8, 1.0)
plt.ylabel("Score")
plt.xlabel("")
plt.title("Model Performance Comparison")
plt.legend(title="Metric")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=FIG_DPI)
plt.close()

# ============================
# 2️ PRECISION vs RECALL
# ============================

fig, ax = plt.subplots(figsize=(7, 7))

for _, row in df.iterrows():
    ax.scatter(row["recall"], row["precision"], s=120)
    ax.annotate(
        row["approach"],
        (row["recall"], row["precision"]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9
    )

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Tradeoff")

# Limits with padding
ax.set_xlim(0.84, 1.01)
ax.set_ylim(0.84, 1.01)

# Dense ticks + high precision
ticks = np.arange(0.85, 1.001, 0.01)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# Equal aspect AND centered
ax.set_aspect("equal", adjustable="box")
ax.set_anchor("C")

ax.tick_params(labelsize=10)
ax.title.set_fontsize(14)

ax.grid(True)

plt.savefig(f"{OUTPUT_DIR}/precision_recall.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()



# ============================
# 3️ CONFUSION MATRICES
# ============================

for _, row in df.iterrows():
    cm = np.array(row["confusion_matrix"])

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legit", "Phish"],
        yticklabels=["Legit", "Phish"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(row["approach"])
    plt.tight_layout()

    fname = row["approach"].lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/cm_{fname}.png", dpi=FIG_DPI)
    plt.close()

# ============================
# 4 TABLE FIGURE
# ============================

table_df = df[[
    "approach", "accuracy", "precision", "recall", "f1_score", "auc_roc"
]].copy()

table_df.iloc[:, 1:] = table_df.iloc[:, 1:].round(4)

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/results_table.png", dpi=FIG_DPI)
plt.close()

print(f"Figures saved to: {OUTPUT_DIR}")