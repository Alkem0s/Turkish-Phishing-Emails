import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ============================
# CONFIG
# ============================

RESULTS_PATH = "./results/results.json"
OUTPUT_DIR = "./results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_DPI = 300
STYLE = "whitegrid"

sns.set_theme(style=STYLE)

# ============================
# LOAD RESULTS
# ============================

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

plt.figure(figsize=(6, 6))
for _, row in df.iterrows():
    plt.scatter(row["recall"], row["precision"], s=120)
    plt.text(
        row["recall"] + 0.002,
        row["precision"] + 0.002,
        row["approach"],
        fontsize=9
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Tradeoff")
plt.xlim(0.85, 1.0)
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precision_recall.png", dpi=FIG_DPI)
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

# ============================
# 5️ FEW-SHOT DEGRADATION CURVE
# ============================

DEGRADATION_DIR = "./results/few_shot_analysis"

if os.path.exists(DEGRADATION_DIR):
    degradation_files = sorted([
        f for f in os.listdir(DEGRADATION_DIR)
        if f.startswith("degradation_") and f.endswith(".json")
    ])

    if degradation_files:
        # Load the most recent run
        with open(os.path.join(DEGRADATION_DIR, degradation_files[-1]), "r") as f:
            degradation = json.load(f)

        rows = []
        for exp in degradation["experiments"].values():
            rows.append({
                "samples": exp["sample_size"],
                "TF-IDF": exp["tfidf"]["f1_score"],
                "XLM-R": exp["xlmr"]["f1_score"]
            })

        deg_df = pd.DataFrame(rows).sort_values("samples")

        plt.figure(figsize=(6, 4))
        plt.plot(
            deg_df["samples"],
            deg_df["TF-IDF"],
            marker="o",
            linewidth=2,
            label="TF-IDF + LR"
        )
        plt.plot(
            deg_df["samples"],
            deg_df["XLM-R"],
            marker="s",
            linewidth=2,
            label="XLM-R"
        )

        plt.xlabel("Number of Turkish Training Samples")
        plt.ylabel("F1 Score")
        plt.title("Few-Shot Performance Degradation")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/few_shot_degradation.png",
            dpi=FIG_DPI
        )
        plt.close()

        print("Few-shot degradation figure saved.")
    else:
        print("No degradation result files found.")
else:
    print("Few-shot analysis directory not found.")