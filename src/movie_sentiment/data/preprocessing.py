from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

STEM_SUFFIXES = ("ingly", "edly", "ingly", "ing", "edly", "edly", "ed", "ly", "es", "s")


@dataclass
class TextPreprocessor:
    stopwords: set[str] = field(default_factory=lambda: set(ENGLISH_STOP_WORDS))
    contractions: dict[str, str] = field(default_factory=lambda: DEFAULT_CONTRACTIONS.copy())
    min_token_length: int = 2

    def _expand_contractions(self, text: str) -> str:
        for src, target in self.contractions.items():
            text = text.replace(src, target)
        return text

    @staticmethod
    def _simple_stem(token: str) -> str:
        if len(token) <= 4:
            return token
        for suffix in STEM_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    def clean(self, text: str, stem: bool = True, remove_stops: bool = True) -> str:
        raw = str(text or "").lower().strip()
        raw = self._expand_contractions(raw)
        raw = re.sub(r"\s+", " ", raw)
        raw = raw.translate(str.maketrans("", "", string.punctuation))

        tokens: list[str] = []
        for token in raw.split():
            token = token.strip()
            if len(token) < self.min_token_length:
                continue
            if remove_stops and token in self.stopwords:
                continue
            if stem:
                token = self._simple_stem(token)
            if token:
                tokens.append(token)
        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        return self.clean(text).split()

    def transform_series(
        self,
        series: pd.Series,
        stem: bool = True,
        remove_stops: bool = True,
    ) -> pd.Series:
        return series.fillna("").astype(str).map(
            lambda text: self.clean(text=text, stem=stem, remove_stops=remove_stops)
        )
