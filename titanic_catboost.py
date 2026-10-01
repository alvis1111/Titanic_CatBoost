"""Leakage-safe Titanic feature engineering and the published CatBoost model."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

# Limit native libraries before NumPy/CatBoost are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    CATBOOST_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = exc


DECISION_THRESHOLD = 0.50
FINAL_RANDOM_STATE = 90045

RAW_FEATURE_COLUMNS = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "EmbarkedFilled",
    "Title",
    "CabinDeck",
    "TicketPrefix",
    "AgeBand",
    "FareBand",
    "FamilyGroup",
    "SexPclass",
    "ExactTicket",
    "Surname",
    "FamilyKey",
]

NUMERIC_FEATURES = [
    "AgeFilled",
    "FareFilled",
    "SexEncoded",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "FareLog",
    "FarePerPerson",
    "HasCabin",
    "AgePclass",
    "NameLength",
    "WomenChildPriority",
    "AgeMissing",
    "FareMissing",
    "CabinMissing",
    "TicketGroupSize",
    "FarePerTicketMember",
    "FamilyGroupSize",
    "Mother",
    "Child",
    "AdultMale",
]

MODEL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def validate_raw_features(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    missing = set(RAW_FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Titanic fields: {sorted(missing)}")
    if frame["PassengerId"].duplicated().any():
        raise ValueError("PassengerId contains duplicates")


def _normalise_ticket(ticket: pd.Series, passenger_id: pd.Series) -> pd.Series:
    cleaned = ticket.fillna("").astype(str).str.upper().str.replace(
        r"[^A-Z0-9]", "", regex=True
    )
    fallback = "TICKET_PID_" + passenger_id.astype("int64").astype(str)
    return cleaned.where(cleaned.str.len().gt(0), fallback)


def _normalise_surname(name: pd.Series, passenger_id: pd.Series) -> pd.Series:
    surname = (
        name.fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
    )
    fallback = "SURNAME_PID_" + passenger_id.astype("int64").astype(str)
    return surname.where(surname.str.len().gt(0), fallback)


def _extract_title(name: pd.Series) -> pd.Series:
    title = name.fillna("").astype(str).str.extract(
        r",\s*([^.]*)\.", expand=False
    ).fillna("Rare").str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return title.where(title.isin({"Mr", "Mrs", "Miss", "Master"}), "Rare")


def _ticket_prefix(ticket: pd.Series) -> pd.Series:
    prefix = ticket.fillna("").astype(str).str.upper().map(
        lambda value: re.sub(r"[\d./\s]", "", value)
    )
    return prefix.where(prefix.str.len().gt(0), "NONE")


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fit every imputation, frequency, and bin statistic on training rows only."""

    def __init__(self, ticket_prefix_min_count: int = 8):
        self.ticket_prefix_min_count = ticket_prefix_min_count

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TitanicFeatureEngineer":
        validate_raw_features(X)
        raw = X[RAW_FEATURE_COLUMNS].copy()
        title = _extract_title(raw["Name"])

        observed_age = raw.loc[raw["Age"].notna()].copy()
        observed_title = title.loc[observed_age.index]
        age_table = observed_age.assign(_Title=observed_title)
        self.age_group_medians_ = (
            age_table.groupby(["_Title", "Pclass", "Sex"], observed=True)["Age"]
            .median()
            .to_dict()
        )
        self.age_pclass_medians_ = (
            observed_age.groupby("Pclass", observed=True)["Age"].median().to_dict()
        )
        self.age_overall_median_ = float(observed_age["Age"].median())
        self.fare_pclass_medians_ = (
            raw.groupby("Pclass", observed=True)["Fare"].median().to_dict()
        )
        self.fare_overall_median_ = float(raw["Fare"].median())
        embarked_mode = raw["Embarked"].mode(dropna=True)
        self.embarked_mode_ = str(embarked_mode.iloc[0]) if not embarked_mode.empty else "S"

        exact_ticket = _normalise_ticket(raw["Ticket"], raw["PassengerId"])
        surname = _normalise_surname(raw["Name"], raw["PassengerId"])
        family_size = (
            raw["SibSp"].fillna(0).astype(int)
            + raw["Parch"].fillna(0).astype(int)
            + 1
        )
        family_key = surname + "_" + family_size.astype(str)
        self.ticket_group_size_map_ = {
            str(key): int(value) for key, value in exact_ticket.value_counts().items()
        }
        self.family_group_size_map_ = {
            str(key): int(value) for key, value in family_key.value_counts().items()
        }

        raw_prefix = _ticket_prefix(raw["Ticket"])
        prefix_counts = raw_prefix.value_counts()
        self.frequent_ticket_prefixes_ = sorted(
            prefix_counts[
                prefix_counts >= int(self.ticket_prefix_min_count)
            ].index.astype(str).tolist()
        )

        fare_filled = self._fill_fare(raw)
        try:
            _, fare_edges = pd.qcut(fare_filled, q=4, retbins=True, duplicates="drop")
        except ValueError:
            fare_edges = np.asarray([fare_filled.min(), fare_filled.max()])
        fare_edges = np.unique(np.asarray(fare_edges, dtype=float))
        if len(fare_edges) < 2:
            fare_edges = np.asarray([-np.inf, np.inf], dtype=float)
        else:
            fare_edges[0] = -np.inf
            fare_edges[-1] = np.inf
        self.fare_band_edges_ = fare_edges
        self.fare_band_labels_ = [
            f"Q{index + 1}" for index in range(len(fare_edges) - 1)
        ]
        self.feature_names_in_ = np.asarray(RAW_FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(RAW_FEATURE_COLUMNS)
        return self

    def _fill_age(self, raw: pd.DataFrame, title: pd.Series) -> pd.Series:
        values: list[float] = []
        for index, row in raw.iterrows():
            if pd.notna(row["Age"]):
                values.append(float(row["Age"]))
                continue
            key = (str(title.loc[index]), int(row["Pclass"]), str(row["Sex"]))
            values.append(
                float(
                    self.age_group_medians_.get(
                        key,
                        self.age_pclass_medians_.get(
                            int(row["Pclass"]), self.age_overall_median_
                        ),
                    )
                )
            )
        return pd.Series(values, index=raw.index, dtype=float)

    def _fill_fare(self, raw: pd.DataFrame) -> pd.Series:
        values: list[float] = []
        for _, row in raw.iterrows():
            if pd.notna(row["Fare"]):
                values.append(float(row["Fare"]))
            else:
                values.append(
                    float(
                        self.fare_pclass_medians_.get(
                            int(row["Pclass"]), self.fare_overall_median_
                        )
                    )
                )
        return pd.Series(values, index=raw.index, dtype=float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            [
                "age_group_medians_",
                "fare_pclass_medians_",
                "ticket_group_size_map_",
                "family_group_size_map_",
                "fare_band_edges_",
            ],
        )
        validate_raw_features(X)
        raw = X[RAW_FEATURE_COLUMNS].copy()
        result = pd.DataFrame(index=raw.index)

        title = _extract_title(raw["Name"])
        age_filled = self._fill_age(raw, title)
        fare_filled = self._fill_fare(raw)
        family_size = (
            raw["SibSp"].fillna(0).astype(int)
            + raw["Parch"].fillna(0).astype(int)
            + 1
        )
        exact_ticket = _normalise_ticket(raw["Ticket"], raw["PassengerId"])
        surname = _normalise_surname(raw["Name"], raw["PassengerId"])
        family_key = surname + "_" + family_size.astype(str)
        ticket_group_size = exact_ticket.map(self.ticket_group_size_map_).fillna(1)
        family_group_size = family_key.map(self.family_group_size_map_).fillna(1)
        family_group_size = family_group_size.where(family_size.gt(1), 1)
        prefix = _ticket_prefix(raw["Ticket"])
        prefix = prefix.where(prefix.isin(self.frequent_ticket_prefixes_), "OTHER")

        result["Pclass"] = raw["Pclass"].astype(int).astype(str)
        result["Sex"] = raw["Sex"].fillna("unknown").astype(str)
        result["EmbarkedFilled"] = raw["Embarked"].fillna(self.embarked_mode_).astype(str)
        result["Title"] = title.astype(str)
        result["CabinDeck"] = raw["Cabin"].fillna("U").astype(str).str[0].str.upper()
        result["TicketPrefix"] = prefix.astype(str)
        result["AgeBand"] = pd.cut(
            age_filled,
            bins=[-np.inf, 12, 18, 35, 50, 65, np.inf],
            labels=["Child", "Teen", "Young", "Middle", "Senior", "Elder"],
        ).astype(str)
        result["FareBand"] = pd.cut(
            fare_filled,
            bins=self.fare_band_edges_,
            labels=self.fare_band_labels_,
            include_lowest=True,
        ).astype(str)
        result["FamilyGroup"] = pd.cut(
            family_size,
            bins=[0, 1, 4, 6, np.inf],
            labels=["Solo", "Small", "Medium", "Large"],
        ).astype(str)
        result["SexPclass"] = result["Sex"] + "_P" + result["Pclass"]
        result["ExactTicket"] = exact_ticket.astype(str)
        result["Surname"] = surname.astype(str)
        result["FamilyKey"] = family_key.astype(str)

        result["AgeFilled"] = age_filled
        result["FareFilled"] = fare_filled
        result["SexEncoded"] = raw["Sex"].map({"male": 0, "female": 1}).fillna(-1)
        result["SibSp"] = raw["SibSp"].fillna(0).astype(int)
        result["Parch"] = raw["Parch"].fillna(0).astype(int)
        result["FamilySize"] = family_size.astype(int)
        result["IsAlone"] = family_size.eq(1).astype(int)
        result["FareLog"] = np.log1p(fare_filled.clip(lower=0))
        result["FarePerPerson"] = fare_filled / family_size.clip(lower=1)
        result["HasCabin"] = raw["Cabin"].notna().astype(int)
        result["AgePclass"] = age_filled * raw["Pclass"].astype(float)
        result["NameLength"] = raw["Name"].fillna("").astype(str).str.len()
        result["WomenChildPriority"] = (
            raw["Sex"].eq("female") | age_filled.lt(16)
        ).astype(int)
        result["AgeMissing"] = raw["Age"].isna().astype(int)
        result["FareMissing"] = raw["Fare"].isna().astype(int)
        result["CabinMissing"] = raw["Cabin"].isna().astype(int)
        result["TicketGroupSize"] = ticket_group_size.astype(float)
        result["FarePerTicketMember"] = fare_filled / ticket_group_size.clip(lower=1)
        result["FamilyGroupSize"] = family_group_size.astype(float)
        result["Mother"] = (
            raw["Sex"].eq("female")
            & raw["Parch"].fillna(0).gt(0)
            & age_filled.ge(18)
            & title.ne("Miss")
        ).astype(int)
        result["Child"] = age_filled.lt(16).astype(int)
        result["AdultMale"] = (raw["Sex"].eq("male") & age_filled.ge(16)).astype(int)

        result = result[MODEL_FEATURE_COLUMNS].copy()
        if result.isna().any().any():
            missing = result.isna().sum()
            raise AssertionError(
                f"Engineered features contain missing values: "
                f"{missing[missing.gt(0)].to_dict()}"
            )
        if not np.isfinite(result[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
            raise AssertionError("Engineered numerical features contain infinity")
        return result

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(MODEL_FEATURE_COLUMNS, dtype=object)


def engineer_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_raw = train.drop(columns=["Survived"], errors="ignore").copy()
    test_raw = test.drop(columns=["Survived"], errors="ignore").copy()
    engineer = TitanicFeatureEngineer().fit(train_raw)
    return (
        engineer.transform(train_raw),
        engineer.transform(test_raw),
        {
            "ticket_group_size_map": dict(engineer.ticket_group_size_map_),
            "family_group_size_map": dict(engineer.family_group_size_map_),
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
        },
    )


class TitanicCatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        iterations: int = 500,
        depth: int = 5,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 5.0,
        random_strength: float = 0.5,
        random_state: int = FINAL_RANDOM_STATE,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "TitanicCatBoostClassifier":
        if not CATBOOST_AVAILABLE or CatBoostClassifier is None:
            raise ImportError(
                "CatBoost is required. Run: python -m pip install -r requirements.txt"
            ) from CATBOOST_IMPORT_ERROR
        self.feature_engineer_ = TitanicFeatureEngineer().fit(X)
        transformed = self.feature_engineer_.transform(X)
        for column in CATEGORICAL_FEATURES:
            transformed[column] = transformed[column].astype(str)
        categorical_indices = [
            transformed.columns.get_loc(column) for column in CATEGORICAL_FEATURES
        ]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            loss_function="Logloss",
            eval_metric="Accuracy",
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=self.random_state,
        )
        self.model_.fit(
            transformed,
            np.asarray(y, dtype=int),
            cat_features=categorical_indices,
        )
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.n_features_in_ = len(RAW_FEATURE_COLUMNS)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["feature_engineer_", "model_"])
        transformed = self.feature_engineer_.transform(X)
        for column in CATEGORICAL_FEATURES:
            transformed[column] = transformed[column].astype(str)
        return np.asarray(self.model_.predict_proba(transformed), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= DECISION_THRESHOLD).astype(int)


def build_catboost_model() -> TitanicCatBoostClassifier:
    return TitanicCatBoostClassifier()


__all__ = [
    "CATBOOST_AVAILABLE",
    "DECISION_THRESHOLD",
    "FINAL_RANDOM_STATE",
    "RAW_FEATURE_COLUMNS",
    "TitanicCatBoostClassifier",
    "TitanicFeatureEngineer",
    "build_catboost_model",
    "engineer_features",
    "validate_raw_features",
]
