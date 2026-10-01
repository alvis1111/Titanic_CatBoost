from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from make_submission import run_catboost
from titanic_catboost import MODEL_FEATURE_COLUMNS, engineer_features


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION_SHA256 = (
    "a1430b27beb42c1df95ddc99f73f901cb16481d663940518b1b56bda6272b4cd"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CatBoostPackageTests(unittest.TestCase):
    def test_01_published_submission_contract(self) -> None:
        submission = pd.read_csv(ROOT / "data" / "submission_catboost.csv")
        self.assertEqual(submission.shape, (418, 2))
        self.assertListEqual(submission.columns.tolist(), ["PassengerId", "Survived"])
        self.assertListEqual(submission["PassengerId"].tolist(), list(range(892, 1310)))
        self.assertSetEqual(set(submission["Survived"]), {0, 1})
        self.assertEqual(int(submission["Survived"].sum()), 143)
        self.assertEqual(
            sha256_file(ROOT / "data" / "submission_catboost.csv"),
            EXPECTED_SUBMISSION_SHA256,
        )

    @unittest.skipUnless(
        (ROOT / "data" / "raw" / "train.csv").is_file()
        and (ROOT / "data" / "raw" / "test.csv").is_file(),
        "Download Kaggle train.csv and test.csv into data/raw for the E2E test",
    )
    def test_02_training_reproduces_published_submission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="titanic_catboost_test_") as temp:
            output_root = Path(temp)
            summary = run_catboost(
                ROOT / "data" / "raw" / "train.csv",
                ROOT / "data" / "raw" / "test.csv",
                output_root,
            )
            actual = pd.read_csv(output_root / "submission_catboost.csv")
            expected = pd.read_csv(ROOT / "data" / "submission_catboost.csv")
            assert_frame_equal(actual, expected, check_exact=True)
            self.assertEqual(summary["status"], "PASS")
            self.assertTrue(summary["official_source_hashes_verified"])
            self.assertEqual(summary["predicted_survivors"], 143)
            self.assertEqual(
                summary["submission_sha256"], EXPECTED_SUBMISSION_SHA256
            )
            disk_summary = json.loads(
                (output_root / "catboost_run_summary.json").read_text()
            )
            self.assertEqual(disk_summary, summary)

    @unittest.skipUnless(
        (ROOT / "data" / "raw" / "train.csv").is_file()
        and (ROOT / "data" / "raw" / "test.csv").is_file(),
        "Download Kaggle train.csv and test.csv into data/raw for leakage tests",
    )
    def test_03_feature_engineering_is_target_independent(self) -> None:
        train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
        test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
        train_features, test_features, metadata = engineer_features(train, test)
        flipped = train.copy()
        flipped["Survived"] = 1 - flipped["Survived"].astype(int)
        flipped_train, flipped_test, flipped_metadata = engineer_features(flipped, test)
        assert_frame_equal(train_features, flipped_train, check_exact=True)
        assert_frame_equal(test_features, flipped_test, check_exact=True)
        self.assertEqual(metadata, flipped_metadata)
        self.assertNotIn("PassengerId", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("Survived", MODEL_FEATURE_COLUMNS)

    @unittest.skipUnless(
        (ROOT / "data" / "raw" / "train.csv").is_file()
        and (ROOT / "data" / "raw" / "test.csv").is_file(),
        "Download Kaggle train.csv and test.csv into data/raw for leakage tests",
    )
    def test_04_test_rows_do_not_change_training_statistics(self) -> None:
        train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
        test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
        original_train, _, original_metadata = engineer_features(train, test)
        altered = test.copy()
        altered["Ticket"] = [f"UNSEEN-{pid}" for pid in altered["PassengerId"]]
        altered["Name"] = [f"UNSEEN{pid}, Mr. Holdout" for pid in altered["PassengerId"]]
        repeated_train, altered_test, altered_metadata = engineer_features(train, altered)
        assert_frame_equal(original_train, repeated_train, check_exact=True)
        self.assertEqual(original_metadata, altered_metadata)
        self.assertTrue(altered_test["TicketGroupSize"].eq(1).all())

    @unittest.skipUnless(
        (ROOT / "data" / "raw" / "train.csv").is_file()
        and (ROOT / "data" / "raw" / "test.csv").is_file(),
        "Download Kaggle train.csv and test.csv into data/raw for the source-hash test",
    )
    def test_05_modified_source_data_is_rejected(self) -> None:
        train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
        train.loc[0, "Fare"] = float(train.loc[0, "Fare"]) + 0.01
        with tempfile.TemporaryDirectory(prefix="titanic_catboost_hash_test_") as temp:
            temp_root = Path(temp)
            altered_train = temp_root / "train.csv"
            train.to_csv(altered_train, index=False, lineterminator="\n")
            with self.assertRaisesRegex(ValueError, "train.csv SHA-256"):
                run_catboost(
                    altered_train,
                    ROOT / "data" / "raw" / "test.csv",
                    temp_root / "outputs",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
