from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    output_path: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(results_df: pd.DataFrame, output_path: Path) -> Path:
    if results_df.empty:
        raise ValueError("results_df is empty")

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    melted = results_df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Comparison Across Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
