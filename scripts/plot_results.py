"""
FedMedDx Results Plotter & Figure Generator.
Extracts metrics from results/fedmeddx_experiments.db and generates high-resolution figures:
1. results/loss_convergence.png
2. results/accuracy_f1_comparison.png
3. results/modality_benchmark_summary.png
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = os.path.join("results", "fedmeddx_experiments.db")
OUTPUT_DIR = "results"


def plot_all_figures():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found. Run scripts/run_benchmarks.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT r.run_id, r.modality, r.strategy, r.alpha, m.round_num, m.hospital_id,
           m.loss, m.accuracy, m.f1_macro, m.auc_ovr
    FROM experiment_runs r
    JOIN round_metrics m ON r.run_id = m.run_id
    ORDER BY r.created_at, m.round_num
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No experiment records found in database to plot.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # -------------------------------------------------------------
    # Figure 1: Loss Convergence per Modality and Strategy
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df,
        x="round_num",
        y="loss",
        hue="strategy",
        style="modality",
        markers=True,
        dashes=False,
        errorbar=None,
        linewidth=2.2
    )
    plt.title("FedMedDx Training Loss Convergence Across Communication Rounds", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Communication Round / Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=12, fontweight="bold")
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, "loss_convergence.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved figure: {loss_path}")

    # -------------------------------------------------------------
    # Figure 2: Accuracy & Macro-F1 Progression
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(
        data=df,
        x="round_num",
        y="accuracy",
        hue="strategy",
        style="modality",
        markers=True,
        ax=axes[0],
        linewidth=2.0
    )
    axes[0].set_title("Validation Accuracy per Round", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Communication Round", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)

    sns.lineplot(
        data=df,
        x="round_num",
        y="f1_macro",
        hue="strategy",
        style="modality",
        markers=True,
        ax=axes[1],
        linewidth=2.0
    )
    axes[1].set_title("Macro F1-Score per Round", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Communication Round", fontsize=11)
    axes[1].set_ylabel("Macro F1-Score", fontsize=11)

    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, "accuracy_f1_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved figure: {acc_path}")

    # -------------------------------------------------------------
    # Figure 3: Summary Bar Chart: Local-Only vs FedRep+FedBN
    # -------------------------------------------------------------
    # Get final round metric per run
    final_rounds = df.groupby(["run_id", "hospital_id"]).last().reset_index()
    # Filter to primary comparison: Local-Only vs FedRep+FedBN
    comparison_df = final_rounds[final_rounds["strategy"].isin(["Local-Only", "FedRep+FedBN"])]
    if comparison_df.empty:
        comparison_df = final_rounds

    plt.figure(figsize=(12, 6))
    strategy_palette = {"Local-Only": "#94a3b8", "FedRep+FedBN": "#0284c7", "FedRep": "#0d9488", "FedAvg": "#f59e0b"}
    ax = sns.barplot(
        data=comparison_df,
        x="modality",
        y="accuracy",
        hue="strategy",
        palette=strategy_palette,
        capsize=0.1
    )
    plt.title("Performance Comparison: Isolated Local-Only vs. FedRep Collaborative", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Disease Modality", fontsize=12, fontweight="bold")
    plt.ylabel("Final Convergence Accuracy", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "modality_benchmark_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Saved figure: {summary_path}")


if __name__ == "__main__":
    plot_all_figures()
