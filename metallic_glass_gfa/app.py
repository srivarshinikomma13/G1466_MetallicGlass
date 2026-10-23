"""Streamlit interface for the metallic-glass GFA regression project."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from preprocessing import DATA_PATH, TARGET, engineer_features, load_dataset  # noqa: E402

MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

st.set_page_config(page_title="Metallic Glass GFA Prediction", page_icon="⚗️", layout="wide")
st.title("Metallic Glass Forming Ability Prediction System")


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_preprocessor():
    return joblib.load(MODELS / "preprocessor.pkl")


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    if name == "ANN":
        import tensorflow as tf

        return tf.keras.models.load_model(MODELS / "ann_model.keras")
    path = MODELS / f"{name.lower().replace(' ', '_')}.pkl"
    return joblib.load(path)


def available_models() -> list[str]:
    path = MODELS / "model_metrics.csv"
    if not path.exists():
        return []
    metrics = pd.read_csv(path)
    names = []
    for name in metrics["Model"].astype(str):
        model_file = MODELS / ("ann_model.keras" if name == "ANN" else f"{name.lower().replace(' ', '_')}.pkl")
        if model_file.exists():
            names.append(name)
    return names


def show_image(filename: str, caption: str) -> None:
    path = PLOTS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def prediction_page(dataset: pd.DataFrame, model_names: list[str]) -> None:
    st.header("GFA Prediction")
    if not model_names:
        st.warning("No trained model files are available. Run `python src/train_ml.py` first.")
        return
    selected = st.selectbox("Select Model", model_names)
    config_path = MODELS / "feature_config.json"
    if not config_path.exists():
        st.error("Saved feature configuration is missing. Retrain the models to create it.")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_features = config["raw_features"]
    elements = config["elements"]
    with st.form("gfa_prediction_form"):
        st.subheader("Alloy and thermal inputs")
        inputs: dict[str, object] = {}
        columns = st.columns(2)
        for index, feature in enumerate(raw_features):
            column = columns[index % 2]
            if feature == "Composition_normalized":
                default = str(dataset[feature].dropna().iloc[0]) if feature in dataset else ""
                inputs[feature] = column.text_input(
                    "Normalized composition (element + atomic percent)", value=default,
                    help="Example from this dataset: Al8.3Ni25.0Zr66.7",
                )
            elif feature in {"Family", "Subfamily"}:
                options = sorted(dataset[feature].dropna().astype(str).unique().tolist())
                if not options:
                    options = ["Unknown"]
                inputs[feature] = column.selectbox(feature, options)
            else:
                values = pd.to_numeric(dataset[feature], errors="coerce")
                default = float(values.median()) if values.notna().any() else 0.0
                if not math.isfinite(default):
                    default = 0.0
                inputs[feature] = column.number_input(
                    feature, value=default, format="%.3f", help="Temperature in kelvin."
                )
        submitted = st.form_submit_button("Predict GFA", type="primary", use_container_width=True)

    if submitted:
        from preprocessing import parse_composition

        if not parse_composition(inputs.get("Composition_normalized", "")):
            st.error("Enter a composition containing valid element symbols and positive atomic percentages.")
            return
        try:
            feature_frame = pd.DataFrame([inputs], columns=raw_features)
            engineered = engineer_features(feature_frame, elements)
            preprocessor = get_preprocessor()
            transformed = preprocessor.transform(engineered)
            model = get_model(selected)
            if selected == "ANN":
                prediction = float(model.predict(transformed, verbose=0).reshape(-1)[0])
            else:
                prediction = float(model.predict(transformed)[0])
            st.success(f"Predicted critical casting diameter: **{prediction:.3f} mm**")
            st.caption(f"Model: {selected}. The prediction is a point estimate, not a confidence interval.")
        except FileNotFoundError as error:
            st.error(f"A required model artifact is missing: {error}")
        except Exception as error:
            st.error(f"Prediction could not be completed. Check the input fields and model artifacts. Details: {error}")


def main() -> None:
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Home", "GFA Prediction", "Model Performance", "Dataset Information", "About Project"],
            label_visibility="collapsed",
        )
        models = available_models()
        if models:
            st.caption("Available trained models")
            st.write(", ".join(models))

    try:
        dataset = get_dataset()
    except Exception as error:
        st.error(f"Dataset could not be loaded: {error}")
        st.info("Place the supplied CSV at `data/metallic_glass_dataset.csv` and restart the app.")
        return

    if page == "Home":
        st.header("Metallic Glass Forming Ability Prediction")
        st.write(
            "Metallic glasses are amorphous metallic materials whose properties depend strongly on "
            "their ability to form a glassy structure during cooling. This application uses "
            "regression models to estimate critical casting diameter from composition and "
            "thermal material features."
        )
        first, second, third = st.columns(3)
        target_values = pd.to_numeric(dataset.get(TARGET), errors="coerce")
        first.metric("Dataset records", f"{len(dataset):,}")
        second.metric("Measured diameters", f"{target_values.notna().sum():,}")
        third.metric("Target", "Critical diameter (mm)")
        show_image("target_distribution.png", "Distribution of measured critical casting diameters")
        st.info("Predictions are educational estimates and should not replace experimental validation.")
    elif page == "GFA Prediction":
        prediction_page(dataset, available_models())
    elif page == "Model Performance":
        st.header("Model Performance")
        metrics_path = MODELS / "model_metrics.csv"
        if metrics_path.exists():
            st.dataframe(pd.read_csv(metrics_path).sort_values("RMSE"), use_container_width=True, hide_index=True)
            best_path = MODELS / "best_model.json"
            if best_path.exists():
                best = json.loads(best_path.read_text(encoding="utf-8"))
                st.caption(f"Best machine-learning model: {best['model']} by held-out RMSE ({best['value']:.3f} mm).")
        else:
            st.warning("Evaluation results are not available yet. Run the training pipeline first.")
        for filename, caption in [
            ("model_comparison.png", "Held-out model comparison"),
            ("feature_importance.png", "Random Forest feature importance"),
            ("ann_training_loss.png", "ANN training and validation loss"),
        ]:
            show_image(filename, caption)
    elif page == "Dataset Information":
        st.header("Dataset Information")
        st.write("Source: Zenodo record 22848120, Master_Database sheet. The CSV retains the source table columns.")
        metrics = pd.to_numeric(dataset[TARGET], errors="coerce") if TARGET in dataset else pd.Series(dtype=float)
        a, b, c = st.columns(3)
        a.metric("Records", len(dataset))
        b.metric("Source columns", len(dataset.columns))
        c.metric("Measured target rows", int(metrics.notna().sum()))
        st.write(f"Target: `{TARGET}` (critical casting diameter, mm), continuous regression.")
        st.write("Columns", list(dataset.columns))
        st.write("Missing values", dataset.isna().sum().rename("Missing").to_frame())
        st.dataframe(dataset.head(30), use_container_width=True, hide_index=True)
    else:
        st.header("About Project")
        st.subheader("Objective")
        st.write("Estimate metallic-glass critical casting diameter from measured alloy composition and thermal properties.")
        st.subheader("Dataset and target")
        st.write("The project uses the Zenodo audited multisource BMG database. `dc_mm` is the continuous critical casting diameter in millimetres; rows without this measured target are excluded from training.")
        st.subheader("Models")
        st.write("The available regression models are listed from trained artifacts in the sidebar. ANN is a fully connected network, appropriate for tabular numerical and categorical inputs. CNN is not used because this dataset is tabular rather than image-based.")
        st.subheader("Technologies")
        st.write("Python, Pandas, NumPy, scikit-learn, XGBoost (when installed), TensorFlow/Keras (optional), Streamlit, Matplotlib, Seaborn, and Joblib.")
        st.warning("The source workbook notes that the reconstructed dataset is still subject to audit. Results from this small, multisource dataset are exploratory and should be validated on family-separated or prospective data.")


main()