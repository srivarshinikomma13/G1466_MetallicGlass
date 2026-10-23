"""Dataset loading and leakage-safe feature preparation for GFA regression."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "metallic_glass_dataset.csv"
TARGET = "dc_mm"
RAW_FEATURES = [
    "Composition_normalized",
    "Family",
    "Subfamily",
    "Tg_K",
    "Tx_K",
    "Tm_K",
    "Tl_K",
]
TEMPERATURE_FEATURES = ["Tg_K", "Tx_K", "Tm_K", "Tl_K"]
CATEGORICAL_FEATURES = ["Family", "Subfamily"]
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the project CSV, or the source workbook's master sheet if supplied."""
    if not path.exists():
        workbook = path.with_name("source_database.xlsx")
        if workbook.exists():
            return pd.read_excel(workbook, sheet_name="Master_Database")
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place metallic_glass_dataset.csv in data/."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Master_Database")
    return pd.read_csv(path)


def parse_composition(value: Any) -> dict[str, float]:
    """Read element atomic percentages from strings such as Al8.3Ni25.0Zr66.7."""
    if pd.isna(value):
        return {}
    matches = ELEMENT_PATTERN.findall(str(value).replace(" ", ""))
    amounts = {element: float(amount) for element, amount in matches}
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {element: amount / total for element, amount in amounts.items()}


def engineer_features(frame: pd.DataFrame, elements: list[str] | None = None) -> pd.DataFrame:
    """Build thermal, family, and composition-fraction predictors."""
    missing = [name for name in RAW_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model input columns: {', '.join(missing)}")

    result = frame[RAW_FEATURES].copy()
    for name in TEMPERATURE_FEATURES:
        result[name] = pd.to_numeric(result[name], errors="coerce")

    parsed = result["Composition_normalized"].map(parse_composition)
    if elements is None:
        elements = sorted({element for composition in parsed for element in composition})
    result["Number_of_elements"] = parsed.map(lambda values: len(values) if values else np.nan)
    for element in elements:
        result[f"atomic_fraction_{element}"] = parsed.map(
            lambda values: values.get(element, 0.0) if values else np.nan
        )
    return result.drop(columns="Composition_normalized")


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def describe_dataset(frame: pd.DataFrame, target: str = TARGET) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "data_types": {key: str(value) for key, value in frame.dtypes.items()},
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "target": target,
        "target_rows": int(pd.to_numeric(frame[target], errors="coerce").notna().sum())
        if target in frame
        else 0,
    }


def save_feature_config(elements: list[str], raw_features: list[str] = RAW_FEATURES) -> None:
    output = PROJECT_ROOT / "models" / "feature_config.json"
    output.write_text(
        json.dumps(
            {"target": TARGET, "raw_features": raw_features, "elements": elements},
            indent=2,
        ),
        encoding="utf-8",
    )