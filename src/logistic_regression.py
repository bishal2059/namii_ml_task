import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest,SelectFromModel, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score

# --- Helper: Clean DataFrame ---
def clean_df(df, id_col='ID'):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    df = df.drop_duplicates(subset=id_col)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Clip outliers to 1st and 99th percentile
    for col in numeric_cols:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)
    return df

# --- 1. Load and Clean Training Data ---
train_df = pd.read_csv('../train_set.csv')
train_df = clean_df(train_df)
X = train_df.drop(columns=['ID', 'CLASS'])
y = train_df['CLASS']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# --- 2. Preprocessing & Feature Selection ---
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('to_float32', FunctionTransformer(lambda x: x.astype(np.float32)))
])

feature_selection = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=1e-4)),
    ('select_k', SelectKBest(score_func=f_classif,k=2500)),
    # ('tree_based', SelectFromModel(ExtraTreesClassifier(n_estimators=50))),
    # ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
])

# --- 3. Build & Fit Model  ---
pipeline = Pipeline([
    ('preprocess', preprocessing),
    ('feature_sel', feature_selection),
    ('clf', LogisticRegression(max_iter=10000))
])

# Grid Search for Hyperparameter Tuning
grid_params = {
    'feature_sel__select_k__k': [500,1000,1500, all],  # Number of features to select
    'feature_sel__select_k__score_func': [f_classif, mutual_info_classif],  # Feature selection methods
    'feature_sel__var_thresh__threshold': [1e-4, 1e-3, 1e-2, 1e-1,1],  # Variance threshold
    'clf__penalty': ['l2'],       
    'clf__C': [0.01, 0.1, 1, 10],                
    'clf__solver': ['saga','lbfgs'],                                          
    'clf__max_iter': [10000],                   
}

grid_search = GridSearchCV(
    pipeline, grid_params,
    cv=2, scoring='roc_auc', n_jobs=1, verbose=2
)
# Fit directly on the training set
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# --- 4. Validation Metrics ---
y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]
metrics_val = {
    'Accuracy': accuracy_score(y_val, y_val_pred),
    'AUROC': roc_auc_score(y_val, y_val_proba),
    'Sensitivity': recall_score(y_val, y_val_pred),
    'Specificity': recall_score(y_val, y_val_pred, pos_label=0),
    'F1-score': f1_score(y_val, y_val_pred)
}
print("Validation metrics:")
for m, v in metrics_val.items():
    print(f"{m}: {v:.4f}")

# --- 5. Test Set Evaluation ---
test_df = pd.read_csv('../test_set.csv')
test_df = clean_df(test_df)
X_test = test_df.drop(columns=['ID', 'CLASS'])
y_test = test_df['CLASS']
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
metrics_test = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'AUROC': roc_auc_score(y_test, y_test_proba),
    'Sensitivity': recall_score(y_test, y_test_pred),
    'Specificity': recall_score(y_test, y_test_pred, pos_label=0),
    'F1-score': f1_score(y_test, y_test_pred)
}
print("Test metrics:")
for m, v in metrics_test.items():
    print(f"{m}: {v:.4f}")


#save best model hyperparameters to CSV
best_params_df = pd.DataFrame(grid_search.best_params_, index=[0])
best_params_df.to_csv('../results/best_params_logistic.csv', index=False)
print("Best hyperparameters saved to '../results/best_params_logistic.csv'.")

#Save test metrics to CSV
metrics_df = pd.DataFrame(metrics_test, index=[0])
metrics_df.to_csv('../results/test_metrics_logistic.csv', index=False)
print("Test metrics saved to '../results/test_metrics_logistic.csv'.")

# --- 6.Evaluate on Blinded Set ---
blinded_df = pd.read_csv('../blinded_test_set.csv')
X_blinded = blinded_df.drop(columns=['ID'])
y_blinded_pred = best_model.predict(X_blinded)

#Save predictions to CSV
predictions_df = pd.DataFrame({
    'ID': blinded_df['ID'],
    'CLASS': y_blinded_pred
})
predictions_df.to_csv('../results/blinded_predictions_logistic.csv', index=False)
print("Blinded predictions saved to '../blinded_predictions_logistic.csv'.")
