from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


@dataclass
class Evaluator:
    average: str = "macro"
    zero_division: int = 0

    def score(self, y_true, y_pred) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)
            ),
            "recall": float(recall_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)),
            "f1": float(f1_score(y_true, y_pred, average=self.average, zero_division=self.zero_division)),
        }

    def report(self, y_true, y_pred, label_names: list[str] | None = None) -> str:
        return classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=self.zero_division,
        )

    def compare_predictions(self, truth, predictions: dict[str, object]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for model_name, y_pred in predictions.items():
            metrics = self.score(truth, y_pred)
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
