"""Train an optional tabular ANN on the same saved split as the ML models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import DATA_PATH, TARGET, engineer_features, load_dataset  # noqa: E402


def main() -> None:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is unavailable in this environment. The ML pipeline and app remain usable; "
            "install a compatible TensorFlow build or use WSL/Linux to train the ANN."
        ) from error

    models = ROOT / "models"
    plots = ROOT / "plots"
    data = load_dataset(DATA_PATH).drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.loc[data[TARGET].notna() & (data[TARGET] > 0)].reset_index(drop=True)
    split = json.loads((models / "split_indices.json").read_text(encoding="utf-8"))
    train_ids, test_ids = split["train"], split["test"]
    config = json.loads((models / "feature_config.json").read_text(encoding="utf-8"))
    features = engineer_features(data, config["elements"])
    preprocessor = joblib.load(models / "preprocessor.pkl")
    X_train = preprocessor.transform(features.iloc[train_ids]).astype(np.float32)
    X_test = preprocessor.transform(features.iloc[test_ids]).astype(np.float32)
    y_train = data[TARGET].iloc[train_ids].to_numpy(dtype=np.float32)
    y_test = data[TARGET].iloc[test_ids].to_numpy(dtype=np.float32)

    tf.keras.utils.set_random_seed(42)
    network = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    network.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    history = network.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    prediction = network.predict(X_test, verbose=0).reshape(-1)
    results = {
        "Model": "ANN",
        "MAE": float(mean_absolute_error(y_test, prediction)),
        "MSE": float(mean_squared_error(y_test, prediction)),
        "RMSE": float(mean_squared_error(y_test, prediction) ** 0.5),
        "R2": float(r2_score(y_test, prediction)),
    }
    network.save(models / "ann_model.keras")
    metrics_path = models / "model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["Model"] != "ANN"]
    pd.concat([metrics, pd.DataFrame([results])], ignore_index=True).to_csv(metrics_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history.history["loss"], label="Training loss")
    axis.plot(history.history["val_loss"], label="Validation loss")
    axis.set(title="ANN Training and Validation Loss", xlabel="Epoch", ylabel="Mean squared error (mm²)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "ann_training_loss.png", dpi=160)
    plt.close(figure)
    print(f"ANN held-out metrics: {results}")
    print(f"Epochs completed: {len(history.history['loss'])}")


if __name__ == "__main__":
    main()