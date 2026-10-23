# Prediction of Glass Forming Ability of Metallic Glasses

## Project Summary

This project predicts the continuous critical casting diameter (`dc_mm`, millimetres) of a metallic glass from its normalized composition, chemical family, and measured thermal properties. A larger critical diameter generally indicates that a thicker section can be cast amorphous under the reported experimental conditions. It is a regression problem, not a classification task.

The data are from the **Audited Multisource Database of Bulk Metallic Glass Forming Ability** on Zenodo, record [10.5281/zenodo.22848120](https://doi.org/10.5281/zenodo.22848120), licensed CC BY 4.0. The source workbook's `Master_Database` sheet is exported here to `data/metallic_glass_dataset.csv`. Its 276 rows and 23 columns contain 239 measured `dc_mm` values and 37 missing targets. Training uses only rows with a positive measured `dc_mm`. The source's `Include_model` flags and README indicate that the reconstruction/audit is evolving; consequently, results are exploratory and not a materials-design guarantee.

## Dataset Fields

The source fields are preserved in the CSV. Model inputs are exactly `Composition_normalized`, `Family`, `Subfamily`, `Tg_K`, `Tx_K`, `Tm_K`, and `Tl_K`. The composition is parsed into measured elemental atomic fractions and a constituent-element count. Temperature fields are numeric and are imputed only using training-set medians. Family fields are imputed and one-hot encoded using training data. IDs, DOI, source notes, QC flags, inclusion flags, and target-derived fields are excluded from predictors.

## System Architecture

`Master_Database` workbook → CSV → positive measured target rows → composition feature engineering → train/test split → train-only imputation/scaling/encoding → regression models → held-out metrics and plots → saved artifacts → Streamlit input form and prediction.

The split is a reproducible random 80/20 holdout (seed 42). Because the data contain related alloy families and literature sources, this split is not a test of transfer to entirely unseen chemical families. Family-aware/prospective validation is needed before scientific deployment.

## Installation

Use Python 3.11 or newer, then from this project directory run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow does not install on native Windows, install the supported TensorFlow build in WSL/Linux or omit TensorFlow and run the classical ML pipeline; the app will list only trained models.

## Train and Run

From `metallic_glass_gfa/`:

```powershell
python src/train_ml.py
python src/train_ann.py
python src/evaluate.py
streamlit run app.py
```

`train_ml.py` fits five regression models, saves training-only preprocessing and individual models, and produces EDA/feature plots. `train_ann.py` trains the optional dense neural network on the same saved split and preprocessing. `evaluate.py` ranks all available results by held-out RMSE, saves the best classical model to `models/best_model.pkl`, and generates the model comparison plot. The app normally opens at `http://localhost:8501`.

The supplied CSV is ready to use. For a different source file, place a compatible CSV at `data/metallic_glass_dataset.csv` with the model input fields above and continuous `dc_mm`; missing required fields are reported rather than guessed.

## Models and Evaluation

Classical models: Linear Regression, Decision Tree, Random Forest, Support Vector Regression, and XGBoost. The ANN is a fully connected tabular network (128, 64, and 32 ReLU units with dropout and a linear output). ANN is appropriate for numeric/categorical tabular material data; no CNN is used because this dataset contains no image or spatial structure input.

Regression metrics are MAE, MSE, RMSE (primary selection metric), and R², calculated on the held-out test partition. The completed classical run (seed 42, 20% holdout, 239 measured rows) yielded:

| Model | MAE (mm) | RMSE (mm) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 2.393 | 4.416 | 0.704 |
| XGBoost | 2.613 | 4.700 | 0.665 |
| Support Vector Regression | 3.150 | 5.333 | 0.569 |
| Random Forest | 3.267 | 5.938 | 0.466 |
| Decision Tree | 3.705 | 7.063 | 0.244 |

Linear Regression had the lowest measured holdout RMSE in this split. This is not evidence of prospective or unseen-family superiority. TensorFlow installation on this Windows environment failed during package extraction because of the Windows long-path limit; ANN training was attempted but did not run, and no ANN result/model is claimed. The classical app and pipeline run without TensorFlow.

## Streamlit Application

The sidebar provides Home, GFA Prediction, Model Performance, Dataset Information, and About Project. Prediction inputs match the saved training feature schema. Use a normalized composition such as `Al8.3Ni25.0Zr66.7`, select family labels, and provide temperatures in kelvin. The output is a model point estimate in millimetres, not a confidence interval.

## Artifacts

- `models/preprocessor.pkl`: training-fitted imputation, scaling, and category encoder.
- `models/feature_config.json` and `models/feature_names.pkl`: exact input schema and composition elements.
- `models/model_metrics.csv`: calculated holdout results.
- `models/best_model.pkl`: best classical model by RMSE.
- `models/ann_model.keras`: optional ANN model when TensorFlow training succeeds.
- `plots/`: target distribution, correlation, feature importance, model comparison, scatter, and optional ANN loss plots.

## Limitations

The database is multisource, small relative to modern ML datasets, and its own documentation describes ongoing reconstruction/audit work. Experimental conditions and chemical-family effects can limit transferability. Missing targets are not fabricated. Random holdout scores can be optimistic when related materials occur in both partitions. Treat predictions as research/education estimates and validate experimentally or with family-separated prospective data.

## Future Work

Use a frozen, independently audited cohort; compare grouped and leave-one-family-out validation; add processing/nucleation metadata when reliably available; quantify uncertainty using a validated method; and assess prospective external data.