from __future__ import annotations

import numpy as np
import pandas as pd


def map_labels_for_scheme(y_series: pd.Series, scheme: str = "five") -> pd.Series:
    scheme = scheme.lower().strip()

    if scheme == "five":
        return y_series.astype(int)

    if scheme == "three":
        mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        return y_series.map(mapping)

    if scheme == "two":
        mapping = {0: 0, 1: 0, 2: np.nan, 3: 1, 4: 1}
        return y_series.map(mapping)

    raise ValueError(f"Unsupported label scheme: {scheme}")


def apply_scheme(df: pd.DataFrame, label_col: str = "Sentiment", scheme: str = "five") -> pd.DataFrame:
    out = df.copy()
    out["Label"] = map_labels_for_scheme(out[label_col], scheme=scheme)
    out = out.dropna(subset=["Label"]).copy()
    out["Label"] = out["Label"].astype(int)
    out["Scheme"] = scheme
    return out
