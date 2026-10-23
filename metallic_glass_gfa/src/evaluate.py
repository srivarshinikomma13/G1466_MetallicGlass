"""Rank trained regressors and produce the comparison plot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    metrics_path = MODELS / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run src/train_ml.py before evaluating models.")
    results = pd.read_csv(metrics_path)
    if results.empty:
        raise ValueError("No model evaluation results are available.")
    results = results.sort_values("RMSE", ascending=True).reset_index(drop=True)
    results.to_csv(metrics_path, index=False)
    winner = str(results.iloc[0]["Model"])
    model_path = MODELS / f"{winner.lower().replace(' ', '_')}.pkl"
    shutil.copy2(model_path, MODELS / "best_model.pkl")
    (MODELS / "best_model.json").write_text(
        json.dumps({"model": winner, "metric": "RMSE", "value": float(results.iloc[0]["RMSE"])}, indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    results.sort_values("RMSE", ascending=False).plot.barh(x="Model", y="RMSE", ax=axis, legend=False, color="#287a72")
    axis.set(title="Model Comparison on Held-Out Data", xlabel="Root mean squared error (mm; lower is better)", ylabel="Model")
    figure.tight_layout()
    figure.savefig(ROOT / "plots" / "model_comparison.png", dpi=160)
    plt.close(figure)
    print(results.to_string(index=False))
    print(f"Best model by held-out RMSE: {winner}")
    print(f"Saved as {MODELS / 'best_model.pkl'}")


if __name__ == "__main__":
    main()