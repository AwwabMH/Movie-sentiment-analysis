from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from movie_sentiment.config import TransformerConfig
from movie_sentiment.utils.io import ensure_dir
from movie_sentiment.utils.seed import set_global_seed


def _require_transformer_stack():
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise ImportError(
            "Transformer pipeline requires torch and transformers packages. "
            "Install dependencies from requirements.txt."
        ) from exc


class PhraseDataset:
    def __init__(self, encodings: dict[str, np.ndarray], labels: np.ndarray):
        import torch

        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


@dataclass
class WeightedProbabilityEnsembler:
    def blend(self, probabilities: list[np.ndarray], weights: list[float]) -> np.ndarray:
        if not probabilities:
            raise ValueError("No probability tensors were provided for blending")
        if len(probabilities) != len(weights):
            raise ValueError("Each probability tensor must have a matching weight")

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value")

        stacked = np.zeros_like(probabilities[0], dtype=np.float64)
        for prob, weight in zip(probabilities, weights):
            stacked += prob * weight
        return stacked / total_weight


class TransformerFamilyTrainer:
    """Cross-validated transformer trainer with weighted model-family ensembling."""

    def __init__(self, config: TransformerConfig, output_dir: Path):
        self.config = config
        self.output_dir = ensure_dir(output_dir)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    def _split_indices(
        self,
        df: pd.DataFrame,
        label_col: str,
        group_col: str,
    ):
        labels = df[label_col].to_numpy()
        if self.config.use_group_kfold and group_col in df.columns:
            groups = df[group_col].to_numpy()
            splitter = GroupKFold(n_splits=self.config.n_splits)
            return splitter.split(np.zeros(len(df)), labels, groups)

        splitter = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        return splitter.split(np.zeros(len(df)), labels)

    def _train_single_fold(
        self,
        model_name: str,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str],
        val_labels: np.ndarray,
        num_classes: int,
        fold_output_dir: Path,
    ) -> np.ndarray:
        _require_transformer_stack()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        train_enc = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_len,
        )
        val_enc = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_len,
        )

        train_ds = PhraseDataset(train_enc, train_labels)
        val_ds = PhraseDataset(val_enc, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

        args = TrainingArguments(
            output_dir=str(fold_output_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=25,
            seed=self.config.random_state,
            fp16=(self.config.use_mixed_precision and torch.cuda.is_available()),
            report_to=[],
        )

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()

        logits = trainer.predict(val_ds).predictions
        probs = self._softmax(np.asarray(logits))
        return probs

    def train_model_cv(
        self,
        model_name: str,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        group_col: str,
        num_classes: int,
        model_output_dir: Path,
    ) -> dict[str, object]:
        set_global_seed(self.config.random_state)
        records = df.reset_index(drop=True).copy()

        oof_probs = np.zeros((len(records), num_classes), dtype=np.float64)
        y_true = records[label_col].to_numpy(dtype=int)

        for fold, (train_idx, val_idx) in enumerate(self._split_indices(records, label_col, group_col), start=1):
            fold_dir = ensure_dir(model_output_dir / f"fold_{fold}")

            train_rows = records.iloc[train_idx]
            val_rows = records.iloc[val_idx]

            val_probs = self._train_single_fold(
                model_name=model_name,
                train_texts=train_rows[text_col].astype(str).tolist(),
                train_labels=train_rows[label_col].to_numpy(dtype=int),
                val_texts=val_rows[text_col].astype(str).tolist(),
                val_labels=val_rows[label_col].to_numpy(dtype=int),
                num_classes=num_classes,
                fold_output_dir=fold_dir,
            )
            oof_probs[val_idx] = val_probs

        y_pred = oof_probs.argmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        return {
            "model_name": model_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "oof_probs": oof_probs,
            "metrics": metrics,
        }

    def train_weighted_ensemble(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        group_col: str,
    ) -> dict[str, object]:
        if df.empty:
            raise ValueError("Input dataframe for transformer training is empty")

        n_classes = int(df[label_col].nunique())
        if n_classes < 2:
            raise ValueError("At least two classes are required for training")

        family_probs: list[np.ndarray] = []
        weights: list[float] = []
        per_model_results: list[dict[str, object]] = []

        for model_name, weight in self.config.model_specs:
            model_dir = ensure_dir(self.output_dir / model_name.replace("/", "_"))
            result = self.train_model_cv(
                model_name=model_name,
                df=df,
                text_col=text_col,
                label_col=label_col,
                group_col=group_col,
                num_classes=n_classes,
                model_output_dir=model_dir,
            )
            family_probs.append(result["oof_probs"])
            weights.append(float(weight))
            per_model_results.append(result)

        blender = WeightedProbabilityEnsembler()
        ensemble_probs = blender.blend(family_probs, weights)
        y_true = df[label_col].to_numpy(dtype=int)
        y_pred = ensemble_probs.argmax(axis=1)

        ensemble_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "oof_probs": ensemble_probs,
            "metrics": ensemble_metrics,
            "per_model_results": per_model_results,
        }
