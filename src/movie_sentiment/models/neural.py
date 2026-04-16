from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import Sequential  # noqa: F401
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: F401
        from tensorflow.keras.layers import (  # noqa: F401
            LSTM,
            BatchNormalization,
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            Input,
            SpatialDropout1D,
        )
    except Exception as exc:
        raise ImportError("tensorflow is required for neural models") from exc


@dataclass
class KerasNeuralFactory:
    random_state: int = 42

    def build_mlp(self, input_dim: int, num_classes: int = 5):
        _require_tensorflow()
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_bilstm(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_classes: int = 5,
    ):
        _require_tensorflow()
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            LSTM,
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            Input,
            SpatialDropout1D,
        )
        from tensorflow.keras.optimizers import Adam

        model = Sequential(
            [
                Input(shape=(max_seq_len,)),
                Embedding(vocab_size, embed_dim, trainable=True),
                SpatialDropout1D(0.30),
                Bidirectional(LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.20)),
                Bidirectional(LSTM(32, dropout=0.20)),
                Dense(64, activation="relu"),
                Dropout(0.30),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


def keras_predict_classes(model, x) -> np.ndarray:
    probs = model.predict(x, verbose=0)
    return np.argmax(probs, axis=1)
