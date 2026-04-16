from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from movie_sentiment.config import DataConfig
from movie_sentiment.data.preprocessing import TextPreprocessor


@dataclass
class ReviewDatasetLoader:
    config: DataConfig
    preprocessor: TextPreprocessor | None = None

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = {
            self.config.text_column,
            self.config.label_column,
            self.config.group_column,
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

    def load_reviews(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        self._validate_required_columns(df)

        df = df.copy()
        df[self.config.text_column] = df[self.config.text_column].fillna("").astype(str)
        df[self.config.label_column] = df[self.config.label_column].astype(int)

        if self.preprocessor is not None:
            df[self.config.cleaned_text_column] = self.preprocessor.transform_series(df[self.config.text_column])
        elif self.config.cleaned_text_column not in df.columns:
            df[self.config.cleaned_text_column] = df[self.config.text_column]

        return df

    def build_context_column(
        self,
        df: pd.DataFrame,
        phrase_col: str | None = None,
        context_col: str = "ContextText",
    ) -> pd.DataFrame:
        phrase_col = phrase_col or self.config.text_column
        out = df.copy()
        sentence_text = out.groupby(self.config.group_column)[phrase_col].transform(
            lambda s: " ".join(s.astype(str).tolist())
        )
        out[context_col] = out[phrase_col].astype(str) + " [SEP] " + sentence_text.astype(str)
        return out

    def load_phase2_comparison(self) -> pd.DataFrame | None:
        if not self.config.phase2_comparison_path.exists():
            return None
        return pd.read_csv(self.config.phase2_comparison_path)

    def load_final_comparison(self) -> pd.DataFrame | None:
        if not self.config.final_comparison_path.exists():
            return None
        return pd.read_csv(self.config.final_comparison_path)
