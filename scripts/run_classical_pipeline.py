from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    _bootstrap_src_path()

    from movie_sentiment.config import DataConfig
    from movie_sentiment.pipelines.classical_pipeline import ClassicalSentimentPipeline

    parser = argparse.ArgumentParser(description="Run the OOP classical sentiment pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts")
    args = parser.parse_args()

    pipeline = ClassicalSentimentPipeline(data_config=DataConfig(output_dir=args.output_dir))
    results = pipeline.run()

    print("\nTop classical models by macro F1:")
    print(results[["model", "f1", "accuracy"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
