# Credit Card Fraud Detection

An end-to-end machine learning system that detects fraudulent credit card transactions, returns a fraud probability (risk score) for each transaction, and serves predictions through an interactive Streamlit application with a fraud-statistics dashboard.

Fraud is extremely rare in this problem (about 0.17 % of transactions in the dataset), so the project focuses on the challenges that come with that imbalance: leakage-safe preprocessing, resampling, PR-AUC-based evaluation, and decision-threshold tuning instead of the default 0.5 cut-off.

```
Raw Transactions → EDA & Preprocessing → Feature Analysis → Imbalance Handling → Model Training
                 → Evaluation → Optimization & Threshold Tuning → Explainability → Fraud Prediction App
```

## Dataset

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (European cardholders, September 2013): 284,807 transactions over two days, of which 492 are fraudulent. Features `V1`–`V28` are anonymised PCA components; `Time` and `Amount` are the only original features; `Class` is the target (1 = fraud).

Place `creditcard.csv` in `data/raw/`.

## Team & Responsibilities

| Member | Role | Responsibilities | Deliverable |
|---|---|---|---|
| **1** | Data Understanding & Preprocessing | Select and inspect the dataset; understand columns and target; run `info()`, `describe()` and distribution checks; handle missing values, duplicates, outliers and data types; encode and scale features where required; split into X / y and stratified train / test sets; build a basic EDA contrasting fraud and normal transactions; save the cleaned data and a reusable preprocessing pipeline. | Data Understanding + Preprocessing + EDA notebook |
| **2** | Feature Engineering & Imbalanced Data | Quantify the class imbalance; evaluate mitigation strategies (`class_weight`, random under/over-sampling, SMOTE where appropriate); engineer and select features; run correlation analysis; measure the impact of new features; build leakage-free pipelines; compare results before and after imbalance handling. | Feature Engineering + Imbalance Handling notebook with clear comparisons |
| **3** | Machine Learning Models & Evaluation | Train and compare Logistic Regression, Random Forest and XGBoost/LightGBM (plus extra models where suitable); evaluate with Precision, Recall, F1-score, Confusion Matrix, ROC-AUC and PR-AUC; summarise all models in a comparison table and plots; analyse false positives and false negatives; select the candidate model for the final stage. | Model Comparison notebook + Evaluation Report |
| **4** | Optimization, Explainability & Deployment | Tune the best model's hyperparameters (`GridSearchCV` / `RandomizedSearchCV`); select a decision threshold instead of defaulting to 0.5; compare performance before vs after optimization; explain predictions with SHAP / feature importance; persist the final model with `joblib`; build a Streamlit web app where users submit a transaction and receive *Legitimate* or *Potential Fraud* with a fraud probability; add a dashboard of fraud statistics; connect the final model to the app. | Optimized model + explainability analysis + final Streamlit application |

## Repository Structure

```
fraud-detection-project/
├── app/
│   └── streamlit_app.py            # Member 4 - prediction app + dashboard
├── artifacts/                      # Saved models, preprocessor, metrics, figures
│   ├── preprocessor.joblib         # Member 1
│   ├── xgboost_fraud_model.joblib  # Member 3 (baseline)
│   ├── final_fraud_model.joblib    # Member 4 (tuned model + threshold)
│   ├── metrics.json, dashboard_stats.json, sample_transactions.csv
│   └── figures/                    # SHAP, threshold and comparison plots
├── data/
│   ├── raw/creditcard.csv
│   └── processed/                  # X_train / X_test / y_train / y_test (parquet)
├── member2_outputs/                # SMOTE-resampled train set, class weights
├── notebooks/
│   ├── 01_data_understanding_preprocessing.ipynb
│   ├── 02_feature_engineering_imbalance.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_optimization_explainability.ipynb
├── prepare_data.py                 # Rebuilds processed data + preprocessor from the raw CSV
├── requirements.txt
└── README.md
```

> **Note:** `.gitignore` excludes `*.csv`, `*.parquet` and `artifacts/*.joblib`, so datasets and trained models are **not** stored in the repository. Follow *Getting Started* to regenerate them, or get the files from a teammate.

## Getting Started

**1. Create an environment** (Python 3.12 recommended; the scientific stack, `shap` in particular, has the most reliable prebuilt packages there):

```bash
conda create -n fraud python=3.12 -y
conda activate fraud
python -m pip install -r requirements.txt ipykernel
```

Or with `venv`: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`), then the same `pip install`.

**2. Add the data.** Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

**3. Generate the processed data and preprocessor:**

```bash
python prepare_data.py
```

This reproduces Member 1's preprocessing (same split and scalers) and creates `data/processed/*.parquet` and `artifacts/preprocessor.joblib`. Alternatively, run notebook `01`.

**4. Run the notebooks.** Notebooks `02` and `03` are analysis and model comparison. Notebook `04` only needs the outputs of step 3: it tunes the model, selects the threshold, produces the SHAP plots and writes everything the app needs to `artifacts/`. Select the `fraud` kernel when opening it.

**5. Launch the app** from the project root:

```bash
streamlit run app/streamlit_app.py
```

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` in a notebook | The kernel uses a different Python than the one you installed into. In a notebook cell run `%pip install <package>`, or pick the right kernel. |
| `pyarrow ... binary incompatibility` | Restart the kernel; if it persists: `python -m pip install --upgrade --force-reinstall numpy pandas pyarrow`. |
| `FileNotFoundError: ../data/processed/X_train.parquet` | The processed data was never generated (it is not in git). Run `python prepare_data.py`. |
| App shows "Missing artifact" | Run notebook `04` first so `artifacts/final_fraud_model.joblib` exists. |

## Using the App

- **Predict**: choose a real example transaction, enter values manually, or upload a CSV for batch scoring. The app returns the verdict, the fraud probability, a risk level, and the features that pushed the score up or down. The decision threshold can be adjusted from the sidebar.
- **Fraud Dashboard**: overall fraud statistics, activity and fraud rate by hour of day, and amount statistics per class.
- **Model & Explainability**: before/after comparison, chosen threshold, SHAP plots and confusion matrices.

## Key Design Decisions

- **Leakage prevention**: the split is done before any fitting; scalers are fitted on the training set only; SMOTE is applied inside a cross-validation pipeline during tuning so validation folds never contain synthetic samples.
- **Metric choice**: PR-AUC drives model selection and tuning because accuracy and ROC-AUC are misleading at this imbalance level.
- **Threshold**: chosen from out-of-fold training predictions (F1-optimal, or a minimum-recall rule) and never from the test set, which is used once for the final report.
- **Explainability**: SHAP values explain the model globally and per transaction.

## Results

Fill in from `artifacts/metrics.json` after running notebook 04.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline XGBoost (threshold 0.5) | | | | |
| Tuned XGBoost (threshold 0.5) | | | | |
| Tuned XGBoost + optimized threshold | | | | |

## Limitations & Future Work

- The dataset is anonymised and two days long, so performance may not transfer to other periods or institutions; models need periodic retraining as fraud patterns drift.
- The best threshold depends on the real cost of a missed fraud versus a false alarm; a cost-sensitive threshold could replace the F1 rule.
- Possible extensions: LightGBM / stacking, probability calibration, a REST API (FastAPI) in front of the model, and Docker deployment.
