"""Verify the public CatBoost package without needing Kaggle raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED_SUBMISSION_SHA256 = "a1430b27beb42c1df95ddc99f73f901cb16481d663940518b1b56bda6272b4cd"
REQUIRED_PACKAGE_FILES = (
    "README.md",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    "requirements.txt",
    "manifest.json",
    "SHA256SUMS",
    "titanic_catboost.py",
    "make_submission.py",
    "verify_package.py",
    "tests/test_contract.py",
    "data/raw/README.md",
    "reports/catboost_result.json",
    "reports/model_comparison.csv",
    "docs/kaggle_public_score.jpg",
)
REQUIRED_CHECKSUM_PATHS = {
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "README.md",
    "data/raw/README.md",
    "data/submission_catboost.csv",
    "docs/kaggle_public_score.jpg",
    "make_submission.py",
    "manifest.json",
    "outputs/.gitkeep",
    "reports/catboost_result.json",
    "reports/model_comparison.csv",
    "requirements.txt",
    "tests/__init__.py",
    "tests/test_contract.py",
    "titanic_catboost.py",
    "verify_package.py",
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".matplotlib",
    ".pytest_cache",
    "__MACOSX",
    "__pycache__",
    "catboost_info",
    "kaggle.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_checksum_manifest() -> int:
    checksum_path = ROOT / "SHA256SUMS"
    require(checksum_path.is_file(), "Missing SHA256SUMS")
    checked: set[str] = set()
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (\./[^\r\n]+)", raw_line)
        require(match is not None, f"Malformed SHA256SUMS line {line_number}")
        expected, relative_text = match.groups()
        relative = Path(relative_text[2:])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe checksum path")
        normalized = relative.as_posix()
        require(normalized != "SHA256SUMS", "SHA256SUMS cannot checksum itself")
        require(normalized not in checked, f"Duplicate checksum entry: {normalized}")
        target = (ROOT / normalized).resolve()
        require(target.is_relative_to(ROOT.resolve()), f"Checksum path escapes package: {normalized}")
        require(target.is_file() and not target.is_symlink(), f"Checksum target missing: {normalized}")
        require(sha256_file(target) == expected, f"Checksum mismatch: {normalized}")
        checked.add(normalized)
    missing = sorted(REQUIRED_CHECKSUM_PATHS - checked)
    require(not missing, f"SHA256SUMS is missing required entries: {missing}")
    return len(checked)


def verify_static_contract(allowed_raw_names: set[str] | None = None) -> dict[str, object]:
    allowed_raw_names = allowed_raw_names or set()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    frozen = manifest["frozen_submission"]
    require(
        frozen["sha256"] == EXPECTED_SUBMISSION_SHA256,
        "Manifest frozen submission SHA-256 was changed",
    )
    submission_path = ROOT / frozen["path"]
    require(submission_path.is_file(), f"Missing {frozen['path']}")
    submission = pd.read_csv(submission_path)
    require(submission.shape == (418, 2), f"Submission shape={submission.shape}")
    require(
        submission.columns.tolist() == ["PassengerId", "Survived"],
        f"Submission columns={submission.columns.tolist()}",
    )
    require(
        submission["PassengerId"].tolist() == list(range(892, 1310)),
        "PassengerId must be ordered 892..1309",
    )
    require(not submission.isna().any().any(), "Submission contains missing values")
    require(set(submission["Survived"]) == {0, 1}, "Survived is not binary")
    require(int(submission["Survived"].sum()) == frozen["predicted_survivors"], "Survivor count mismatch")
    require(sha256_file(submission_path) == frozen["sha256"], "Submission SHA-256 mismatch")

    missing = [relative for relative in REQUIRED_PACKAGE_FILES if not (ROOT / relative).is_file()]
    require(not missing, f"Missing package files: {missing}")
    checksum_count = verify_checksum_manifest()

    junk = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.name in FORBIDDEN_NAMES or path.suffix in {".pyc", ".pyo"}
    ]
    require(not junk, f"Forbidden generated/private files: {junk}")
    raw_csv = {
        path.name for path in (ROOT / "data" / "raw").glob("*.csv")
    }
    unexpected_raw = sorted(raw_csv - allowed_raw_names)
    require(not unexpected_raw, f"Public package contains Kaggle raw data: {unexpected_raw}")

    sensitive_markers = (b"/Users/", b"xwechat_files", b"/private/tmp", b"kaggle.json\"")
    text_suffixes = {".md", ".py", ".json", ".txt", ".csv", ""}
    leaked: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        content = path.read_bytes()
        if any(marker in content for marker in sensitive_markers):
            leaked.append(str(path.relative_to(ROOT)))
    require(not leaked, f"Absolute/private marker found: {leaked}")
    return {
        "submission_sha256": frozen["sha256"],
        "rows": len(submission),
        "predicted_survivors": int(submission["Survived"].sum()),
        "files_checked": len(REQUIRED_PACKAGE_FILES),
        "checksums_verified": checksum_count,
    }


def verify_retraining(train_path: Path, test_path: Path) -> None:
    from make_submission import run_catboost

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    require(
        sha256_file(train_path) == manifest["source_sha256"]["train.csv"],
        "train.csv SHA-256 mismatch",
    )
    require(
        sha256_file(test_path) == manifest["source_sha256"]["test.csv"],
        "test.csv SHA-256 mismatch",
    )
    with tempfile.TemporaryDirectory(prefix="catboost_verify_") as temp:
        summary = run_catboost(train_path, test_path, temp)
        require(
            summary["submission_sha256"]
            == manifest["frozen_submission"]["sha256"],
            "Retrained prediction does not match the frozen submission",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--strict-public",
        action="store_true",
        help="Fail if data/raw contains CSV files; use this before publishing.",
    )
    args = parser.parse_args()
    if (args.train is None) != (args.test is None):
        parser.error("--train and --test must be provided together")

    default_train = (ROOT / "data" / "raw" / "train.csv").resolve()
    default_test = (ROOT / "data" / "raw" / "test.csv").resolve()
    if args.train is None and not args.strict_public:
        if default_train.exists() != default_test.exists():
            parser.error("data/raw must contain both train.csv and test.csv")
        if default_train.exists():
            args.train, args.test = default_train, default_test

    train_path = args.train.resolve() if args.train is not None else None
    test_path = args.test.resolve() if args.test is not None else None
    uses_local_raw = (
        not args.strict_public
        and train_path == default_train
        and test_path == default_test
    )
    allowed_raw = {"train.csv", "test.csv"} if uses_local_raw else set()
    result = verify_static_contract(allowed_raw_names=allowed_raw)
    label = "Package contract with local raw data" if uses_local_raw else "Public package contract"
    print(f"[PASS] {label}")
    print(json.dumps(result, indent=2))
    if train_path is not None and test_path is not None:
        verify_retraining(train_path, test_path)
        print("[PASS] Exact CatBoost retraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
