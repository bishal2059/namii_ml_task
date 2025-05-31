## NAAMI ML TASK

This repository contains code and results for training, validating, and testing three supervised classification models (Logistic Regression, Random Forest, and XGBoost) on the given dataset. Each model pipeline script resides in the `src/` directory, and their outputs (hyperparameter tuning results, evaluation metrics, and blinded predictions) are saved in the `results/` directory.

All code is written in Python 3 and leverages common machine-learning libraries (scikit-learn, XGBoost, pandas, numpy). The following sections describe how to set up a reproducible environment, run each model pipeline, and locate the generated result files.

---

## Directory Structure

```
naami_ml_task/
├── README.md
├── requirements.txt
├── src/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── train_set.csv
├── test_set.csv
├── blinded_test_set.csv
└── results/
    ├── best_params_logistic.csv
    ├── test_metrics_logistic.csv
    ├── blinded_predictions_logistic.csv
    ├── best_params_random_forest.csv
    ├── test_metrics_random_forest.csv
    ├── blinded_predictions_random_forest.csv
    ├── best_params_xgboost_reduced.csv
    ├── test_metrics_xgboost_reduced.csv
    └── blinded_predictions_xgboost_reduced.csv
```

* **src/**: Contains Python scripts for each model pipeline.
* CSV files (`train_set.csv`, `test_set.csv`, `blinded_test_set.csv`) are located at the project root.
* **results/**: Folder where each script writes its outputs (hyperparameters, validation/test metrics, blinded predictions).
* **requirements.txt**: Lists all Python packages required to run the code.

---

## Setup

1. **Install Python 3** (version 3.7 or higher recommended). Verify with:

   ```bash
   python3 --version
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on macOS/Linux
   .\venv\Scripts\activate     # on Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include at least:

   ```text
   pandas
   numpy
   scikit-learn
   xgboost
   ```

4. **Verify environment** by importing key packages:

   ```bash
   python -c "import pandas; import numpy; import sklearn; import xgboost; print('OK')"
   ```

---

## Running the Pipelines

Each script expects to find the input CSV files at the project root:

* `train_set.csv` (training data with `ID` and `CLASS` columns)
* `test_set.csv` (test data with `ID` and `CLASS` columns)
* `blinded_test_set.csv` (unlabeled data with only `ID`)

The scripts automatically perform data cleaning (median imputation, outlier clipping, drop duplicates), feature selection (variance threshold + SelectKBest), hyperparameter tuning via `GridSearchCV`, and evaluation on both validation and test sets. In the final step, they generate predictions on the blinded test set.

> **Note:** When running from the project root, the scripts use relative paths like `../train_set.csv` because they are located in the `src/` directory. Ensure your working directory is the project root when invoking them.

### 1. Logistic Regression

**Script:** `src/logistic_regression.py`

**Usage:**

```bash
python src/logistic_regression.py
```

**What it does:**

* Reads `train_set.csv`. Cleans data and splits into train/validation.
* Constructs a pipeline that imputes missing values, scales features, selects top-k features via ANOVA F-test (with variance threshold), and fits a regularized Logistic Regression model.
* Performs grid search over:

  * Number of features `k` (e.g., 500, 1000, 1500, all)
  * Variance threshold (e.g., 1e-4, 1e-3, 1e-2, 1e-1, 1)
  * Feature selection method (ANOVA F-test, mutual information)
  * Regularization strength `C`, solver choice
* Outputs:

  * **`results/best_params_logistic.csv`**: Single-row CSV recording the best hyperparameters found.
  * **`results/test_metrics_logistic.csv`**: Single-row CSV containing accuracy, AUROC, sensitivity, specificity, and F1-score on the test set.
  * **`results/blinded_predictions_logistic.csv`**: Two-column CSV (`ID`, `CLASS`) with predicted labels on `blinded_test_set.csv`.

---

### 2. Random Forest

**Script:** `src/random_forest.py`

**Usage:**

```bash
python src/random_forest.py
```

**What it does:**

* Reads `train_set.csv`, cleans, and splits into train/validation.
* Builds a pipeline that imputes missing values, scales data, selects features (variance threshold + SelectKBest), then fits a Random Forest classifier.
* Performs grid search over:

  * Feature selection (`k`, variance threshold, score function)
  * Random Forest parameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`)
* Outputs:

  * **`results/best_params_random_forest.csv`**: Best hyperparameters for Random Forest.
  * **`results/test_metrics_random_forest.csv`**: Accuracy, AUROC, sensitivity, specificity, F1-score on test set.
  * **`results/blinded_predictions_random_forest.csv`**: Predictions for the blinded test set.

---

### 3. XGBoost

**Script:** `src/xgboost_model.py`

**Usage:**

```bash
python src/xgboost_model.py
```

**What it does:**

* Reads `train_set.csv`, cleans, and splits into train/validation.
* Pipeline: impute → scale → variance threshold → SelectKBest → XGBoost classifier.
* Simplified grid search (reduced for faster runs):

  * `k` in { 1000, all}
  * Variance threshold in {1e-4, 1e-2}
  * `n_estimators` in {100, 200, 500}
  * `max_depth` in {3, 6}
  * `learning_rate = {0.01, 0.1}`, `subsample = 1.0`, `colsample_bytree = 1.0`
* Outputs:

  * **`results/best_params_xgboost_reduced.csv`**: Best hyperparameters from the reduced grid.
  * **`results/test_metrics_xgboost_reduced.csv`**: Evaluation metrics on test set.
  * **`results/blinded_predictions_xgboost_reduced.csv`**: Predictions for blinded test set.


---

## Reproducing Results

1. Ensure that your working directory is the project root.
2. Activate the Python virtual environment (see Setup above).
3. Confirm that the CSV files are present at the project root:

   * `train_set.csv`
   * `test_set.csv`
   * `blinded_test_set.csv`
4. Run each model script:

   ```bash
   python src/logistic_regression.py
   python src/random_forest.py
   python src/xgboost_model.py
   ```
5. After each run, check the `results/` directory to verify that the relevant CSVs were generated.

If all runs complete without errors, you will have reproduced the hyperparameter tuning and evaluation results.

---

