from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfTextVectorizer:
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 10_000
    min_df: int = 2

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)


@dataclass
class Word2VecSentenceEmbedder:
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 30
    sg: int = 1
    seed: int = 42

    def fit(self, tokenized_texts: list[list[str]]) -> "Word2VecSentenceEmbedder":
        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            raise ImportError("gensim is required for Word2VecSentenceEmbedder") from exc

        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            epochs=self.epochs,
            sg=self.sg,
            seed=self.seed,
        )
        return self

    def _mean_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        if not hasattr(self, "model"):
            raise RuntimeError("Word2Vec model has not been fitted yet")
        return np.vstack([self._mean_vector(tokens) for tokens in tokenized_texts])


@dataclass
class Doc2VecSentenceEmbedder:
    vector_size: int = 150
    window: int = 5
    min_count: int = 1
    epochs: int = 50
    seed: int = 42

    def fit(self, tokenized_texts: list[list[str]]) -> "Doc2VecSentenceEmbedder":
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except Exception as exc:
            raise ImportError("gensim is required for Doc2VecSentenceEmbedder") from exc

        docs = [TaggedDocument(words=tokens, tags=[f"doc_{idx}"]) for idx, tokens in enumerate(tokenized_texts)]
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            epochs=self.epochs,
            dm=1,
            seed=self.seed,
        )
        self.model.build_vocab(docs)
        self.model.train(docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        if not hasattr(self, "model"):
            raise RuntimeError("Doc2Vec model has not been fitted yet")
        vectors = [self.model.infer_vector(tokens) for tokens in tokenized_texts]
        return np.vstack(vectors)
