"""
train_model.py
==============
Train and compare four classification models for customer churn prediction.

Models
------
1. Logistic Regression  (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

Workflow
--------
- Load SMOTE-balanced training data and original test data.
- For each model: train, cross-validate (5-fold stratified), tune hyperparameters.
- Compare all four on ROC-AUC.
- Save all models, the best model separately, and a comparison CSV.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from lightgbm import LGBMClassifier

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load processed train and test CSVs.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    print("📂  Loading processed data …")
    try:
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Processed data not found. Run data_preprocessing.py first."
        ) from exc

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def evaluate_model_metrics(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Compute standard classification metrics for a trained model.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn compatible estimator.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.

    Returns
    -------
    dict[str, float]
        Dictionary with Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    """
    Train Logistic Regression with GridSearchCV.

    Returns
    -------
    tuple
        (best_model, metrics_dict, best_params, training_time, cv_score)
    """
    print("\n🔹  Training Logistic Regression …")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    lr = LogisticRegression(random_state=config.RANDOM_STATE)
    grid = GridSearchCV(lr, config.LOGISTIC_REGRESSION_PARAMS, cv=skf, scoring="roc_auc", n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best = grid.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring="roc_auc")
    metrics = evaluate_model_metrics(best, X_test, y_test)

    print(f"   ⏱  Training time: {elapsed:.2f}s")
    print(f"   🏆 Best params: {grid.best_params_}")
    print(f"   📊 CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return best, metrics, grid.best_params_, elapsed, cv_scores.mean()


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    """
    Train Random Forest with GridSearchCV.

    Returns
    -------
    tuple
        (best_model, metrics_dict, best_params, training_time, cv_score)
    """
    print("\n🔹  Training Random Forest …")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(rf, config.RANDOM_FOREST_PARAMS, cv=skf, scoring="roc_auc", n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best = grid.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring="roc_auc")
    metrics = evaluate_model_metrics(best, X_test, y_test)

    print(f"   ⏱  Training time: {elapsed:.2f}s")
    print(f"   🏆 Best params: {grid.best_params_}")
    print(f"   📊 CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return best, metrics, grid.best_params_, elapsed, cv_scores.mean()


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    return best, metrics, search.best_params_, elapsed, cv_scores.mean()


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    """
    Train LightGBM with RandomizedSearchCV for efficiency.

    Returns
    -------
    tuple
        (best_model, metrics_dict, best_params, training_time, cv_score)
    """
    print("\n🔹  Training LightGBM …")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    lgb = LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1)
    search = RandomizedSearchCV(
        lgb, config.LIGHTGBM_PARAM_GRID,
        n_iter=20, cv=skf, scoring="roc_auc",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring="roc_auc")
    metrics = evaluate_model_metrics(best, X_test, y_test)

    print(f"   ⏱  Training time: {elapsed:.2f}s")
    print(f"   🏆 Best params: {search.best_params_}")
    print(f"   📊 CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return best, metrics, search.best_params_, elapsed, cv_scores.mean()


def train_all_models() -> None:
    """
    Train all four models, compare, select best, and save everything.
    """
    X_train, y_train, X_test, y_test = load_data()

    results = {}
    models = {}

    # ── Train each model ─────────────────────────────────────────
    for name, train_fn in [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),

        ("LightGBM", train_lightgbm),
    ]:
        model, metrics, params, train_time, cv_auc = train_fn(X_train, y_train, X_test, y_test)
        models[name] = model
        results[name] = {**metrics, "CV_ROC_AUC": round(cv_auc, 4), "Train_Time_s": round(train_time, 2)}

    # ── Comparison table ─────────────────────────────────────────
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"
    comparison_df = comparison_df.sort_values("ROC-AUC", ascending=False)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70)

    # ── Save comparison ──────────────────────────────────────────
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    comparison_df.to_csv(config.REPORT_PATHS["model_comparison"])
    print(f"\n💾  Saved comparison table → {config.REPORT_PATHS['model_comparison']}")

    # ── Select best model ────────────────────────────────────────
    best_name = comparison_df["ROC-AUC"].idxmax()
    best_model = models[best_name]
    print(f"\n🏆  Best model: {best_name} (ROC-AUC = {comparison_df.loc[best_name, 'ROC-AUC']:.4f})")

    # ── Save all models ──────────────────────────────────────────
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
    }
    for name, model in models.items():
        path = config.MODEL_PATHS[model_key_map[name]]
        joblib.dump(model, path)
        print(f"💾  Saved {name} → {path}")

    joblib.dump(best_model, config.MODEL_PATHS["best_model"])
    joblib.dump(best_name, config.MODEL_PATHS["best_model_name"])
    print(f"💾  Saved best model ({best_name}) → {config.MODEL_PATHS['best_model']}")

    print("\n🎉  Model training complete!\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all_models()
