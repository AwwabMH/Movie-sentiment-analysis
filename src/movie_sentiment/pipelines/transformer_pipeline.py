from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from movie_sentiment.config import DataConfig, TransformerConfig
from movie_sentiment.data.augmentation import DataAugmentor, load_external_sentiment_rows
from movie_sentiment.data.loader import ReviewDatasetLoader
from movie_sentiment.data.preprocessing import TextPreprocessor
from movie_sentiment.data.schemes import apply_scheme
from movie_sentiment.models.transformer import TransformerFamilyTrainer
from movie_sentiment.utils.io import ensure_dir, save_dataframe
from movie_sentiment.utils.seed import set_global_seed


def derive_v1_decisions(comparison_df: pd.DataFrame | None) -> dict[str, object]:
    if comparison_df is None or comparison_df.empty:
        return {
            "best_variant": "hybrid_augmented",
            "best_scheme": "five",
            "class_merging_helped": True,
        }

    required = {"variant", "scheme", "f1"}
    if not required.issubset(set(comparison_df.columns)):
        return {
            "best_variant": "hybrid_augmented",
            "best_scheme": "five",
            "class_merging_helped": True,
        }

    ranked = comparison_df.sort_values("f1", ascending=False).reset_index(drop=True)
    top_row = ranked.iloc[0]

    best_five = comparison_df[comparison_df["scheme"] == "five"]["f1"].max()
    best_merge = comparison_df[comparison_df["scheme"].isin(["three", "two"])]["f1"].max()

    if pd.isna(best_five):
        best_five = -np.inf
    if pd.isna(best_merge):
        best_merge = -np.inf

    return {
        "best_variant": str(top_row["variant"]),
        "best_scheme": str(top_row["scheme"]),
        "class_merging_helped": bool(best_merge > best_five),
    }


def stratified_row_cap(df_in: pd.DataFrame, label_col: str, cap: int, seed: int = 42) -> pd.DataFrame:
    if cap <= 0 or len(df_in) <= cap:
        return df_in.copy()

    df = df_in.copy()
    buckets: list[pd.DataFrame] = []
    per_class_cap = max(1, cap // df[label_col].nunique())

    for _, chunk in df.groupby(label_col):
        take_n = min(len(chunk), per_class_cap)
        buckets.append(chunk.sample(n=take_n, random_state=seed))

    out = pd.concat(buckets, ignore_index=False)
    if len(out) < cap:
        needed = cap - len(out)
        remaining = df.drop(index=out.index, errors="ignore")
        if not remaining.empty:
            extra = remaining.sample(n=min(needed, len(remaining)), random_state=seed)
            out = pd.concat([out, extra], ignore_index=False)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass
class AdvancedTransformerPipeline:
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: TransformerConfig = field(default_factory=TransformerConfig)

    def run(
        self,
        variant: str | None = None,
        schemes: list[str] | None = None,
        export_filename: str = "final_pipeline_comparison_refactored.csv",
    ) -> pd.DataFrame:
        set_global_seed(self.model_config.random_state)

        output_dir = ensure_dir(self.data_config.output_dir / "transformer")
        preprocessor = TextPreprocessor()
        loader = ReviewDatasetLoader(config=self.data_config, preprocessor=preprocessor)
        augmentor = DataAugmentor(seed=self.model_config.random_state)

        base_df = loader.load_reviews()
        base_df = base_df[["Phrase", "Sentiment", self.data_config.group_column, self.data_config.cleaned_text_column]].copy()

        v1_df = loader.load_phase2_comparison()
        decisions = derive_v1_decisions(v1_df)

        active_variant = variant or str(decisions["best_variant"])
        if schemes is None:
            if decisions["class_merging_helped"]:
                active_schemes = [s for s in self.model_config.scheme_names]
            else:
                active_schemes = ["five"]
        else:
            active_schemes = [s.lower().strip() for s in schemes]

        external_rows = load_external_sentiment_rows(max_each=1200, seed=self.model_config.random_state)
        variant_df = augmentor.select_dataset_variant(
            base_rows=base_df,
            variant_name=active_variant,
            external_rows=external_rows,
        )

        if self.data_config.group_column not in variant_df.columns:
            variant_df[self.data_config.group_column] = np.arange(len(variant_df), dtype=int)
        variant_df[self.data_config.group_column] = (
            variant_df[self.data_config.group_column].fillna(-1).astype(int)
        )

        cleaned_col = self.data_config.cleaned_text_column
        if cleaned_col not in variant_df.columns:
            variant_df[cleaned_col] = ""
        missing_clean = variant_df[cleaned_col].isna() | (variant_df[cleaned_col].astype(str).str.strip() == "")
        if missing_clean.any():
            variant_df.loc[missing_clean, cleaned_col] = preprocessor.transform_series(
                variant_df.loc[missing_clean, "Phrase"]
            )

        rows: list[dict[str, object]] = []
        for scheme in active_schemes:
            scheme_df = apply_scheme(variant_df, label_col="Sentiment", scheme=scheme)
            scheme_df = stratified_row_cap(
                scheme_df,
                label_col="Label",
                cap=self.model_config.max_rows_per_variant,
                seed=self.model_config.random_state,
            )

            min_class_count = int(scheme_df["Label"].value_counts().min())
            if min_class_count < 2:
                rows.append(
                    {
                        "pipeline": f"Advanced Transformer Ensemble [{scheme}] on {active_variant}",
                        "accuracy": np.nan,
                        "f1": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "scheme": scheme,
                        "variant": active_variant,
                        "status": "skipped_insufficient_class_rows",
                    }
                )
                continue

            trainer_dir = ensure_dir(output_dir / active_variant / scheme)
            trainer = TransformerFamilyTrainer(config=self.model_config, output_dir=trainer_dir)

            try:
                result = trainer.train_weighted_ensemble(
                    df=scheme_df,
                    text_col=self.data_config.cleaned_text_column,
                    label_col="Label",
                    group_col=self.data_config.group_column,
                )
                metrics = result["metrics"]

                rows.append(
                    {
                        "pipeline": f"Advanced Transformer Ensemble [{scheme}] on {active_variant}",
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "scheme": scheme,
                        "variant": active_variant,
                        "status": "ok",
                    }
                )
            except ImportError as exc:
                rows.append(
                    {
                        "pipeline": f"Advanced Transformer Ensemble [{scheme}] on {active_variant}",
                        "accuracy": np.nan,
                        "f1": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "scheme": scheme,
                        "variant": active_variant,
                        "status": f"dependency_missing: {exc}",
                    }
                )

        results = pd.DataFrame(rows)

        legacy_final = loader.load_final_comparison()
        if legacy_final is not None and not legacy_final.empty:
            if "pipeline" in legacy_final.columns:
                baseline_rows = legacy_final[
                    legacy_final["pipeline"].astype(str).str.contains("Plain BERT Baseline", case=False, na=False)
                ].copy()
                if not baseline_rows.empty:
                    baseline_rows["scheme"] = baseline_rows["pipeline"].str.extract(r"\[(.*?)\]").fillna("unknown")
                    baseline_rows["variant"] = "legacy"
                    baseline_rows["status"] = "legacy_reference"
                    keep_cols = ["pipeline", "accuracy", "f1", "precision", "recall", "scheme", "variant", "status"]
                    baseline_rows = baseline_rows[keep_cols]
                    results = pd.concat([results, baseline_rows], ignore_index=True)

        results = results.sort_values(["f1", "accuracy"], ascending=False, na_position="last").reset_index(drop=True)
        save_dataframe(results, output_dir / export_filename)
        return results


def run_transformer_pipeline(output_dir: Path | None = None) -> pd.DataFrame:
    data_cfg = DataConfig(output_dir=output_dir or Path("outputs"))
    pipeline = AdvancedTransformerPipeline(data_config=data_cfg)
    return pipeline.run()
