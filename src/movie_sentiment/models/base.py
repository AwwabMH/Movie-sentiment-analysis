from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelPrediction:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None = None


@dataclass
class ModelArtifact:
    name: str
    estimator: Any
    train_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
