from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LABEL_MAP_FIVE = {
    0: "Negative",
    1: "Somewhat Negative",
    2: "Neutral",
    3: "Somewhat Positive",
    4: "Positive",
}

LABEL_MAP_THREE = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

LABEL_MAP_TWO = {
    0: "Negative",
    1: "Positive",
}


@dataclass(frozen=True)
class DataConfig:
    data_path: Path = Path("movieReviews.csv")
    phase2_comparison_path: Path = Path("phase2_model_comparison.csv")
    final_comparison_path: Path = Path("final_pipeline_comparison.csv")
    output_dir: Path = Path("outputs")
    text_column: str = "Phrase"
    cleaned_text_column: str = "Phrase_Cleaned"
    label_column: str = "Sentiment"
    group_column: str = "SentenceId"


@dataclass
class ClassicalConfig:
    random_state: int = 42
    test_size: float = 0.20
    tfidf_max_features: int = 10_000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    baseline_c: float = 5.0
    svm_c: float = 10.0
    use_class_weight: bool = True
    include_xgboost: bool = True
    n_jobs: int = -1


@dataclass
class TransformerConfig:
    random_state: int = 42
    model_specs: list[tuple[str, float]] = field(
        default_factory=lambda: [
            ("distilbert-base-uncased", 0.60),
            ("bert-base-uncased", 0.40),
        ]
    )
    scheme_names: list[str] = field(default_factory=lambda: ["five", "three", "two"])
    n_splits: int = 5
    max_len: int = 128
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_rows_per_variant: int = 12_000
    use_group_kfold: bool = True
    use_mixed_precision: bool = True
