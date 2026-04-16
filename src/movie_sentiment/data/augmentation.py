from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd


def load_external_sentiment_rows(max_each: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Load optional external sentiment rows from Hugging Face datasets.

    The function fails gracefully when datasets package or internet access is unavailable.
    """
    random.seed(seed)
    rows: list[dict[str, object]] = []

    try:
        from datasets import load_dataset

        imdb = load_dataset("imdb", split=f"train[:{max_each}]")
        for item in imdb:
            mapped = 4 if int(item["label"]) == 1 else 0
            rows.append({"Phrase": item["text"], "Sentiment": mapped, "source": "imdb"})

        tweet = load_dataset("tweet_eval", "sentiment", split=f"train[:{max_each}]")
        tmap = {0: 0, 1: 2, 2: 4}
        for item in tweet:
            rows.append(
                {
                    "Phrase": item["text"],
                    "Sentiment": tmap.get(int(item["label"]), 2),
                    "source": "tweet_eval",
                }
            )
    except Exception:
        return pd.DataFrame(columns=["Phrase", "Sentiment", "source"])

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass
class DataAugmentor:
    seed: int = 42
    synonym_map: dict[str, list[str]] = field(
        default_factory=lambda: {
            "good": ["great", "solid", "decent"],
            "bad": ["poor", "weak", "awful"],
            "movie": ["film", "feature"],
            "story": ["plot", "narrative"],
            "fun": ["enjoyable", "entertaining"],
            "boring": ["dull", "slow"],
            "excellent": ["superb", "outstanding"],
            "terrible": ["horrible", "awful"],
        }
    )

    def __post_init__(self) -> None:
        random.seed(self.seed)

    @staticmethod
    def random_swap(words: list[str]) -> list[str]:
        if len(words) < 3:
            return words
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        return words

    @staticmethod
    def random_drop(words: list[str], p: float = 0.08) -> list[str]:
        kept = [w for w in words if random.random() > p]
        return kept if kept else words

    def synonym_replace(self, words: list[str], repl_prob: float = 0.20) -> list[str]:
        out: list[str] = []
        for word in words:
            candidates = self.synonym_map.get(word.lower(), [])
            if candidates and random.random() < repl_prob:
                out.append(random.choice(candidates))
            else:
                out.append(word)
        return out

    def perturb_text(self, text: str) -> str:
        words = str(text).split()
        if not words:
            return ""
        words = self.synonym_replace(words, repl_prob=0.25)
        if random.random() < 0.40:
            words = self.random_swap(words)
        words = self.random_drop(words, p=0.10)
        return " ".join(words)

    def build_synthetic_rows(
        self,
        base_rows: pd.DataFrame,
        text_col: str = "Phrase",
        label_col: str = "Sentiment",
        multiplier: float = 1.0,
    ) -> pd.DataFrame:
        if base_rows.empty or multiplier <= 0:
            return pd.DataFrame(columns=[text_col, label_col, "source"])

        synth_size = max(1, int(len(base_rows) * multiplier))
        sampled = base_rows.sample(n=synth_size, replace=True, random_state=self.seed)

        out = sampled[[text_col, label_col]].copy()
        out[text_col] = out[text_col].map(self.perturb_text)
        out["source"] = "synthetic"
        return out

    def select_dataset_variant(
        self,
        base_rows: pd.DataFrame,
        variant_name: str,
        external_rows: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        variant = variant_name.lower().strip()
        base = base_rows.copy()

        if variant == "original_full":
            return base

        if variant == "inliers_only":
            lengths = base["Phrase"].astype(str).str.len()
            cap = lengths.quantile(0.98)
            return base[lengths <= cap].copy()

        if variant == "synthetic_augmented":
            synth = self.build_synthetic_rows(base)
            synth = synth.rename(columns={"source": "variant_source"})
            base["variant_source"] = "base"
            return pd.concat([base, synth], ignore_index=True)

        if variant == "hybrid_augmented":
            synth = self.build_synthetic_rows(base, multiplier=0.7)
            synth = synth.rename(columns={"source": "variant_source"})

            extra = external_rows.copy() if external_rows is not None else pd.DataFrame()
            if not extra.empty:
                if "variant_source" not in extra.columns:
                    extra["variant_source"] = extra.get("source", "external")

            base["variant_source"] = "base"
            chunks = [base, synth]
            if not extra.empty:
                chunks.append(extra[["Phrase", "Sentiment", "variant_source"]])
            return pd.concat(chunks, ignore_index=True)

        raise ValueError(f"Unsupported variant name: {variant_name}")
