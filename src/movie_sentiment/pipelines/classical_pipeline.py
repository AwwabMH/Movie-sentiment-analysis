from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from movie_sentiment.config import ClassicalConfig, DataConfig, LABEL_MAP_FIVE
from movie_sentiment.data.loader import ReviewDatasetLoader
from movie_sentiment.data.preprocessing import TextPreprocessor
from movie_sentiment.evaluation.metrics import Evaluator
from movie_sentiment.evaluation.plots import plot_confusion_matrix, plot_model_comparison
from movie_sentiment.features.vectorizers import TfidfTextVectorizer
from movie_sentiment.models.classical import ClassicalSuiteRunner
from movie_sentiment.utils.io import ensure_dir, save_dataframe
from movie_sentiment.utils.seed import set_global_seed


@dataclass
class ClassicalSentimentPipeline:
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ClassicalConfig = field(default_factory=ClassicalConfig)

    def run(self, export_filename: str = "phase2_model_comparison_refactored.csv") -> pd.DataFrame:
        set_global_seed(self.model_config.random_state)

        output_dir = ensure_dir(self.data_config.output_dir / "classical")
        preprocessor = TextPreprocessor()
        loader = ReviewDatasetLoader(config=self.data_config, preprocessor=preprocessor)

        df = loader.load_reviews()
        text_col = self.data_config.cleaned_text_column
        label_col = self.data_config.label_column

        x_train, x_test, y_train, y_test = train_test_split(
            df[text_col],
            df[label_col],
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
            stratify=df[label_col],
        )

        vectorizer = TfidfTextVectorizer(
            ngram_range=(self.model_config.tfidf_ngram_min, self.model_config.tfidf_ngram_max),
            max_features=self.model_config.tfidf_max_features,
            min_df=self.model_config.tfidf_min_df,
        )
        x_train_vec = vectorizer.fit_transform(x_train.astype(str).tolist())
        x_test_vec = vectorizer.transform(x_test.astype(str).tolist())

        trainer = ClassicalSuiteRunner(self.model_config)
        artifacts = trainer.fit(x_train_vec, y_train.to_numpy())
        predictions = trainer.predict_all(x_test_vec)

        evaluator = Evaluator()
        rows: list[dict[str, object]] = []

        label_names = [LABEL_MAP_FIVE[idx] for idx in sorted(LABEL_MAP_FIVE)]
        for model_name, pred_pack in predictions.items():
            y_pred = pred_pack["y_pred"]
            metrics = evaluator.score(y_test.to_numpy(), y_pred)

            row = {
                "variant": "original_full",
                "scheme": "five",
                "model": model_name,
                "rows": int(len(df)),
                "status": "ok",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f1_original": metrics["f1"],
                "acc_original": metrics["accuracy"],
                "delta_f1_vs_original": 0.0,
                "delta_acc_vs_original": 0.0,
                "train_seconds": round(artifacts[model_name].train_seconds, 3),
            }
            rows.append(row)

            cm_path = output_dir / f"cm_{model_name.replace('+', '_').replace(' ', '_').lower()}.png"
            plot_confusion_matrix(
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                labels=label_names,
                title=f"{model_name} Confusion Matrix",
                output_path=cm_path,
            )

        results_df = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)

        plot_model_comparison(
            results_df=results_df[["model", "accuracy", "precision", "recall", "f1"]],
            output_path=output_dir / "classical_model_comparison.png",
        )

        save_dataframe(results_df, output_dir / export_filename)
        return results_df


def run_classical_pipeline(output_dir: Path | None = None) -> pd.DataFrame:
    data_cfg = DataConfig(output_dir=output_dir or Path("outputs"))
    pipeline = ClassicalSentimentPipeline(data_config=data_cfg)
    return pipeline.run()
