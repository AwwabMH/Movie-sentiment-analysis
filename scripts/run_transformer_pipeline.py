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
    from movie_sentiment.pipelines.transformer_pipeline import AdvancedTransformerPipeline

    parser = argparse.ArgumentParser(description="Run the OOP transformer ensemble pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts")
    parser.add_argument("--variant", type=str, default=None, help="Dataset variant: original_full, inliers_only, synthetic_augmented, hybrid_augmented")
    parser.add_argument(
        "--schemes",
        type=str,
        default=None,
        help="Comma-separated class schemes (five,three,two)",
    )
    args = parser.parse_args()

    scheme_list = None
    if args.schemes:
        scheme_list = [s.strip() for s in args.schemes.split(",") if s.strip()]

    pipeline = AdvancedTransformerPipeline(data_config=DataConfig(output_dir=args.output_dir))
    results = pipeline.run(variant=args.variant, schemes=scheme_list)

    print("\nPipeline results by macro F1:")
    print(results[["pipeline", "f1", "accuracy", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
