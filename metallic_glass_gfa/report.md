# Project Report: Prediction of Glass Forming Ability of Metallic Glasses

## Chapter 1: Introduction

Metallic glasses are metallic alloys with an amorphous, non-crystalline structure. Their formation depends on composition, thermal behavior, and processing. Critical casting diameter is a practical continuous measure of glass-forming ability (GFA): larger measured diameters indicate that a larger cross-section was reported to form glass under the cited experimental conditions.

## Chapter 2: Problem Statement

Estimate critical casting diameter from available alloy composition and material thermal properties. The target is continuous, so this project is formulated as regression.

## Chapter 3: Objectives

- Inspect and preserve the source dataset without creating scientific values.
- Engineer elemental composition fractions from the normalized composition strings.
- Train and compare classical regression models and an optional ANN.
- Save preprocessing and model artifacts for repeatable inference.
- Provide an interactive Streamlit application.

## Chapter 4: Literature and Background

GFA is commonly discussed using experimental critical casting dimensions and thermal descriptors such as glass transition, crystallization, and liquidus temperatures. Different geometric measurements are not automatically interchangeable. This project uses the dataset's explicit `dc_mm` target only and does not substitute thickness or cooling-rate targets.

## Chapter 5: Dataset Description

Source: Aourag, H. (2026), *Audited Multisource Database of Bulk Metallic Glass Forming Ability: Thermal Descriptors, Metric-Clean Critical Diameters, Chemical Families, GFA Cliffs, and Crystallization Metadata*, Zenodo, DOI [10.5281/zenodo.22848120](https://doi.org/10.5281/zenodo.22848120), CC BY 4.0. The workbook's `Master_Database` sheet is exported to `data/metallic_glass_dataset.csv`.

The source sheet contains 276 observations and 23 columns. The continuous target is `dc_mm` in millimetres; 239 rows have measured target values and 37 do not. No duplicate rows were found in the source table. The source workbook notes that its reconstruction/audit status is evolving; the row-level `Include_model` values are inconsistent with the separate final-results cohort description, so the implementation uses the defensible minimal rule of positive measured `dc_mm` and documents this caveat rather than using that flag as a hidden filter.

## Chapter 6: Data Preprocessing

Exact training cohort size, missing-value counts, duplicate count, data types, and model input names are recorded in `models/dataset_summary.json`, `models/run_metadata.json`, and `models/feature_config.json` after training. Rows without a numeric positive target are excluded. Duplicate rows are removed. Missing numeric predictors are median-imputed and standardized; missing categories are mode-imputed and one-hot encoded. Every imputer, scaler, and encoder is fitted on training data only and then applied unchanged to the test data.

## Chapter 7: Feature Engineering

Inputs are normalized composition, chemical family/subfamily, and `Tg_K`, `Tx_K`, `Tm_K`, and `Tl_K`. The composition parser derives one atomic fraction per element present in the training cohort and the number of constituent elements. Element fractions are normalized from the reported composition and do not introduce external or invented material-property constants.

## Chapter 8: Machine Learning Models

Linear Regression, Decision Tree Regressor, Random Forest Regressor, Support Vector Regression, and XGBoost Regressor are evaluated when their dependencies are available. They use the same train/test indices and the same training-fitted transformation.

## Chapter 9: ANN Deep Learning Model

The optional ANN uses dense layers with 128, 64, and 32 ReLU units, dropout, and one linear regression output. It uses Adam, mean squared error, an internal validation split, and early stopping. ANN is suitable as the primary deep-learning experiment for these tabular features. CNN was not used because the dataset is tabular rather than image-based.

## Chapter 10: Experimental Results

The executed classical pipeline used 239 positive measured target rows, an 80/20 random holdout, and seed 42. On the test partition: Linear Regression MAE 2.393 mm, MSE 19.500 mm², RMSE 4.416 mm, R² 0.704; Decision Tree MAE 3.705 mm, MSE 49.891 mm², RMSE 7.063 mm, R² 0.244; Random Forest MAE 3.267 mm, MSE 35.258 mm², RMSE 5.938 mm, R² 0.466; Support Vector Regression MAE 3.150 mm, MSE 28.446 mm², RMSE 5.333 mm, R² 0.569; XGBoost MAE 2.613 mm, MSE 22.087 mm², RMSE 4.700 mm, R² 0.665. TensorFlow could not be imported after its Windows install failed while extracting a long header path; ANN training was attempted but did not execute, so there are no ANN results.

## Chapter 11: Model Comparison

Models are ranked automatically by held-out RMSE in millimetres (lower is better). Linear Regression was the winner in this split, RMSE 4.416 mm. It is selected for demonstration, not claimed as the best generalizing scientific model. MAE, MSE, and R² are also reported. The calculated winner and value are saved to `models/best_model.json`; the comparison graph is saved as `plots/model_comparison.png`.

## Chapter 12: Web Application

`app.py` provides Home, GFA Prediction, Model Performance, Dataset Information, and About Project pages. Prediction controls are created from the saved feature configuration, and model options come only from successfully saved artifacts and computed metrics. Predictions are point estimates, not confidence values. The app's Streamlit health endpoint returned HTTP 200 (`ok`) during execution. A real dataset row with measured diameter 8.0 mm passed through the saved pipeline and produced a predicted value of 8.000 mm.

## Chapter 13: Conclusion

This implementation turns an audited public metallic-glass dataset into a reproducible regression workflow and interactive demonstrator. Conclusions about scientific transferability must remain modest until family-separated and prospective validation are performed.

## Chapter 14: Future Scope

Use a frozen audited dataset version, evaluate leave-one-family-out and source-grouped generalization, add reliable processing/nucleation metadata, and validate uncertainty and performance prospectively.

## Chapter 15: References

1. Aourag, H. (2026). *Audited Multisource Database of Bulk Metallic Glass Forming Ability*. Zenodo. https://doi.org/10.5281/zenodo.22848120
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.