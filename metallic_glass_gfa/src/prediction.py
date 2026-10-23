"""Load saved model artifacts and predict critical casting diameter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict_gfa(input_features: dict[str, Any], model_name: str | None = None) -> float:
    """Predict critical casting diameter (mm) from one row of model inputs."""
    config_path = ROOT / "models" / "feature_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Model feature configuration is missing; train the models first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = config["raw_features"]
    missing = [name for name in required if name not in input_features]
    if missing:
        raise ValueError(f"Missing input values: {', '.join(missing)}")

    frame = pd.DataFrame([{name: input_features[name] for name in required}])
    from preprocessing import engineer_features  # pylint: disable=import-outside-toplevel

    engineered = engineer_features(frame, config["elements"])
    preprocessor = joblib.load(ROOT / "models" / "preprocessor.pkl")
    transformed = preprocessor.transform(engineered)
    if model_name:
        model_file = ROOT / "models" / f"{model_name.lower().replace(' ', '_')}.pkl"
    else:
        model_file = ROOT / "models" / "best_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file is missing: {model_file.name}")
    model = joblib.load(model_file)
    return float(model.predict(transformed)[0])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    dataset_path = ROOT / "data" / "metallic_glass_dataset.csv"
    row = pd.read_csv(dataset_path).dropna(subset=["dc_mm"]).iloc[0]
    payload = {name: row[name] for name in json.loads((ROOT / "models" / "feature_config.json").read_text(encoding="utf-8"))["raw_features"]}
    print(f"Actual diameter: {row['dc_mm']} mm")
    print(f"Predicted diameter: {predict_gfa(payload):.3f} mm")