"""Evaluation utilities for metrics and visual reporting."""

from .metrics import Evaluator
from .plots import plot_confusion_matrix, plot_model_comparison

__all__ = ["Evaluator", "plot_confusion_matrix", "plot_model_comparison"]
