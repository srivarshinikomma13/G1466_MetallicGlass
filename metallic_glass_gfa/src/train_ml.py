"""Train baseline and tree/SVM regressors on the audited GFA dataset."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import (  # noqa: E402
    DATA_PATH,
    PROJECT_ROOT,
    TARGET,
    describe_dataset,
    engineer_features,
    load_dataset,
    make_preprocessor,
    parse_composition,
    save_feature_config,
)


RANDOM_STATE = 42


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    models_dir = ROOT / "models"
    plots_dir = ROOT / "plots"
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    dataset = load_dataset(DATA_PATH)
    if TARGET not in dataset.columns:
        raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")
    dataset = dataset.drop_duplicates().copy()
    dataset[TARGET] = pd.to_numeric(dataset[TARGET], errors="coerce")
    dataset = dataset.loc[dataset[TARGET].notna() & (dataset[TARGET] > 0)].reset_index(drop=True)
    if len(dataset) < 20:
        raise ValueError(f"Only {len(dataset)} measured positive target rows; at least 20 are needed.")

    elements = sorted(
        {
            element
            for value in dataset["Composition_normalized"]
            for element in parse_composition(value)
        }
    )
    features = engineer_features(dataset, elements)
    target = dataset[TARGET].astype(float)
    row_ids = list(range(len(dataset)))
    train_ids, test_ids = train_test_split(
        row_ids, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train = features.iloc[train_ids]
    X_test = features.iloc[test_ids]
    y_train = target.iloc[train_ids]
    y_test = target.iloc[test_ids]

    preprocessor = make_preprocessor(list(features.columns))
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(preprocessor, models_dir / "scaler.pkl")
    save_feature_config(elements)
    joblib.dump(list(features.columns), models_dir / "feature_names.pkl")
    joblib.dump(list(preprocessor.get_feature_names_out()), models_dir / "transformed_feature_names.pkl")
    (models_dir / "split_indices.json").write_text(
        json.dumps({"train": train_ids, "test": test_ids}), encoding="utf-8"
    )

    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=3, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Regression": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
    }
    try:
        from xgboost import XGBRegressor

        estimators["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    except ImportError:
        warnings.warn("XGBoost is not installed; that model will be skipped.", stacklevel=1)

    results: list[dict[str, float | str]] = []
    for name, estimator in estimators.items():
        estimator.fit(X_train_scaled, y_train)
        predictions = estimator.predict(X_test_scaled)
        results.append({"Model": name, **regression_metrics(y_test, predictions)})
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(estimator, models_dir / f"{safe_name}.pkl")
        print(f"{name}: {results[-1]}")

    metrics_path = models_dir / "model_metrics.csv"
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    (models_dir / "dataset_summary.json").write_text(
        json.dumps(describe_dataset(load_dataset(DATA_PATH)), indent=2), encoding="utf-8"
    )
    (models_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": DATA_PATH.name,
                "target": TARGET,
                "target_unit": "mm",
                "target_rows_used": len(dataset),
                "raw_features": list(features.columns),
                "elements": elements,
                "random_state": RANDOM_STATE,
                "test_fraction": 0.2,
                "primary_metric": "RMSE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dataset[TARGET], bins=18, kde=True, ax=ax, color="#287a72")
    ax.set(title="Critical Casting Diameter Distribution", xlabel="Critical casting diameter (mm)", ylabel="Number of alloys")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=160)
    plt.close(fig)

    numeric = features.select_dtypes(include="number").copy()
    numeric[TARGET] = target
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax)
    ax.set(title="Correlation of Numeric Features and Target", xlabel="Feature", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dataset["Tg_K"], dataset[TARGET], alpha=0.7, color="#b45d3b")
    ax.set(title="Glass Transition Temperature vs Critical Diameter", xlabel="Glass transition temperature, Tg (K)", ylabel="Critical casting diameter (mm)")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_scatter.png", dpi=160)
    plt.close(fig)

    forest = estimators["Random Forest"]
    importance = pd.Series(forest.feature_importances_, index=preprocessor.get_feature_names_out())
    importance = importance.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=ax, color="#287a72")
    ax.set(title="Random Forest Feature Importance", xlabel="Impurity-based importance", ylabel="Transformed feature")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    print(f"\nDataset: {len(load_dataset(DATA_PATH))} rows x {len(load_dataset(DATA_PATH).columns)} columns")
    print(f"Measured positive {TARGET} rows used: {len(dataset)}")
    print(f"Raw model features: {list(features.columns)}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()