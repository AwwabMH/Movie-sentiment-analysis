from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from movie_sentiment.config import ClassicalConfig
from movie_sentiment.models.base import ModelArtifact


@dataclass
class ClassicalModelSpec:
    name: str
    estimator: object


class ClassicalSuiteRunner:
    """Train and serve a suite of strong classical baselines."""

    def __init__(self, config: ClassicalConfig):
        self.config = config
        self.artifacts: dict[str, ModelArtifact] = {}

    def _build_specs(self) -> list[ClassicalModelSpec]:
        class_weight = "balanced" if self.config.use_class_weight else None

        specs: list[ClassicalModelSpec] = [
            ClassicalModelSpec(
                name="LR+TFIDF",
                estimator=LogisticRegression(
                    solver="saga",
                    C=self.config.baseline_c,
                    max_iter=2000,
                    class_weight=class_weight,
                    n_jobs=self.config.n_jobs,
                    random_state=self.config.random_state,
                ),
            ),
            ClassicalModelSpec(
                name="SVM+TFIDF",
                estimator=SVC(
                    kernel="rbf",
                    C=self.config.svm_c,
                    gamma="scale",
                    class_weight=class_weight,
                    probability=True,
                    random_state=self.config.random_state,
                ),
            ),
        ]

        if self.config.include_xgboost:
            try:
                from xgboost import XGBClassifier

                specs.append(
                    ClassicalModelSpec(
                        name="XGB+TFIDF",
                        estimator=XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.08,
                            subsample=0.85,
                            colsample_bytree=0.80,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            random_state=self.config.random_state,
                        ),
                    )
                )
            except Exception:
                pass

        return specs

    def fit(self, x_train, y_train) -> dict[str, ModelArtifact]:
        self.artifacts.clear()
        for spec in self._build_specs():
            start = time.perf_counter()
            spec.estimator.fit(x_train, y_train)
            elapsed = time.perf_counter() - start

            self.artifacts[spec.name] = ModelArtifact(
                name=spec.name,
                estimator=spec.estimator,
                train_seconds=elapsed,
                metadata={"type": "classical"},
            )
        return self.artifacts

    def predict_all(self, x_test) -> dict[str, dict[str, np.ndarray | None]]:
        if not self.artifacts:
            raise RuntimeError("No fitted models found. Run fit() first.")

        outputs: dict[str, dict[str, np.ndarray | None]] = {}
        for name, artifact in self.artifacts.items():
            estimator = artifact.estimator
            y_pred = estimator.predict(x_test)

            y_prob = None
            if hasattr(estimator, "predict_proba"):
                y_prob = estimator.predict_proba(x_test)

            outputs[name] = {
                "y_pred": np.asarray(y_pred),
                "y_prob": None if y_prob is None else np.asarray(y_prob),
            }
        return outputs
