"""Train the published Titanic CatBoost candidate and create a Kaggle submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PUBLIC_SCORE_DISPLAYED = 0.79186
FINAL_RANDOM_STATE = 90045


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_catboost_model() -> Any:
    from titanic_catboost import TitanicCatBoostClassifier

    return TitanicCatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        random_state=FINAL_RANDOM_STATE,
    )


def run_catboost(
    train_path: str | Path,
    test_path: str | Path,
    output_root: str | Path,
    verify_source_hashes: bool = True,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    from titanic_catboost import (
        CATBOOST_AVAILABLE,
        DECISION_THRESHOLD,
        RAW_FEATURE_COLUMNS,
    )

    if not CATBOOST_AVAILABLE:
        raise ImportError(
            "CatBoost is required. Run: python -m pip install -r requirements.txt"
        )

    train_path = Path(train_path).expanduser().resolve()
    test_path = Path(test_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_sha256 = sha256_file(train_path)
    test_sha256 = sha256_file(test_path)
    if verify_source_hashes:
        manifest = json.loads(
            (Path(__file__).resolve().parent / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected = manifest["source_sha256"]
        if train_sha256 != expected["train.csv"]:
            raise ValueError(
                "train.csv SHA-256 does not match the official Kaggle file. "
                "Use --allow-nonofficial-data only for an intentional experiment."
            )
        if test_sha256 != expected["test.csv"]:
            raise ValueError(
                "test.csv SHA-256 does not match the official Kaggle file. "
                "Use --allow-nonofficial-data only for an intentional experiment."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if train.shape != (891, 12) or test.shape != (418, 11):
        raise ValueError(f"Unexpected Titanic shapes: train={train.shape}, test={test.shape}")
    if train["PassengerId"].tolist() != list(range(1, 892)):
        raise ValueError("Training PassengerId order must be 1..891")
    if test["PassengerId"].tolist() != list(range(892, 1310)):
        raise ValueError("Test PassengerId order must be 892..1309")

    model = build_catboost_model()
    model.fit(
        train[RAW_FEATURE_COLUMNS],
        train["Survived"].astype(int).to_numpy(),
    )
    probability = np.asarray(
        model.predict_proba(test[RAW_FEATURE_COLUMNS])[:, 1], dtype=float
    )
    if not np.isfinite(probability).all() or not (
        (probability >= 0.0) & (probability <= 1.0)
    ).all():
        raise AssertionError("CatBoost produced invalid probabilities")

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": (probability >= DECISION_THRESHOLD).astype(int),
        }
    )
    submission_path = output_root / "submission_catboost.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")

    import catboost
    import sklearn

    summary: dict[str, object] = {
        "status": "PASS",
        "model": "CatBoost",
        "public_score_displayed": PUBLIC_SCORE_DISPLAYED,
        "decision_threshold": DECISION_THRESHOLD,
        "predicted_survivors": int(submission["Survived"].sum()),
        "model_parameters": {
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "random_seed": FINAL_RANDOM_STATE,
        },
        "source_sha256": {
            "train": train_sha256,
            "test": test_sha256,
        },
        "official_source_hashes_verified": verify_source_hashes,
        "submission_sha256": sha256_file(submission_path),
        "submission_path": "submission_catboost.csv",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
    }
    summary_path = output_root / "catboost_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Titanic exact family/ticket CatBoost submission"
    )
    parser.add_argument("--train", type=Path, default=root / "data" / "raw" / "train.csv")
    parser.add_argument("--test", type=Path, default=root / "data" / "raw" / "test.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs")
    parser.add_argument(
        "--allow-nonofficial-data",
        action="store_true",
        help="Skip official source hashes for an intentional local experiment.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_catboost(
        args.train,
        args.test,
        args.output,
        verify_source_hashes=not args.allow_nonofficial_data,
    )
    output_path = Path(args.output).expanduser().resolve() / str(
        summary["submission_path"]
    )
    print("CatBoost submission complete")
    print(f"Predicted survivors: {summary['predicted_survivors']}")
    print(f"Submission: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
