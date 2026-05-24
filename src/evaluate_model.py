"""
evaluate_model.py
=================
Comprehensive evaluation of the best churn prediction model.

Generates and saves
-------------------
1. Classification report (printed)
2. Confusion matrix heatmap
3. ROC-AUC curve
4. Precision-Recall curve
5. Feature importance plot (top 20)
6. SHAP summary beeswarm plot
7. SHAP bar plot (top 15)
8. SHAP waterfall plots (high / medium / low risk customers)
9. Business-friendly summary
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
})


def load_best_model_and_data():
    """
    Load the best saved model, test data, and feature names.

    Returns
    -------
    tuple
        (model, X_test, y_test, feature_names, model_name)
    """
    print("📂  Loading best model and test data …")
    try:
        model = joblib.load(config.MODEL_PATHS["best_model"])
        model_name = joblib.load(config.MODEL_PATHS["best_model_name"])
        feature_names = joblib.load(config.MODEL_PATHS["feature_names"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Model files not found. Run train_model.py first."
        ) from exc

    try:
        test_df = pd.read_csv(config.TEST_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Test data not found. Run data_preprocessing.py first."
        ) from exc

    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    print(f"   Model: {model_name}  |  Test samples: {len(y_test)}")
    return model, X_test, y_test, feature_names, model_name


# ─────────────────────────────────────────────
# 1. Classification Report
# ─────────────────────────────────────────────
def print_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print a formatted classification report.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))


# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Save a confusion matrix heatmap to reports/.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        linewidths=1, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.REPORT_PATHS["confusion_matrix"], dpi=150)
    plt.close(fig)
    print(f"✅  Saved confusion matrix → {config.REPORT_PATHS['confusion_matrix']}")


# ─────────────────────────────────────────────
# 3. ROC-AUC Curve
# ─────────────────────────────────────────────
def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray) -> None:
    """
    Save an ROC curve with AUC score to reports/.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_proba : np.ndarray
        Predicted probabilities for class 1.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=config.COLORS["primary"], lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=config.COLORS["muted"], lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=config.COLORS["primary"])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.REPORT_PATHS["roc_curve"], dpi=150)
    plt.close(fig)
    print(f"✅  Saved ROC curve → {config.REPORT_PATHS['roc_curve']}")


# ─────────────────────────────────────────────
# 4. Precision-Recall Curve
# ─────────────────────────────────────────────
def plot_precision_recall_curve(y_test: pd.Series, y_proba: np.ndarray) -> None:
    """
    Save a Precision-Recall curve to reports/.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_proba : np.ndarray
        Predicted probabilities for class 1.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=config.COLORS["accent"], lw=2.5, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color=config.COLORS["accent"])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.REPORT_PATHS["pr_curve"], dpi=150)
    plt.close(fig)
    print(f"✅  Saved PR curve → {config.REPORT_PATHS['pr_curve']}")


# ─────────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list[str]) -> None:
    """
    Save a horizontal bar chart of the top-20 feature importances.

    Falls back to model coefficients for linear models.

    Parameters
    ----------
    model : estimator
        Trained model.
    feature_names : list[str]
        Feature names matching model input.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("⚠️  Model does not expose feature importances. Skipping plot.")
        return

    imp_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=config.COLORS["primary"])
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Top 20 Feature Importances", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.REPORT_PATHS["feature_importance"], dpi=150)
    plt.close(fig)
    print(f"✅  Saved feature importance plot → {config.REPORT_PATHS['feature_importance']}")


# ─────────────────────────────────────────────
# 6-8. SHAP Plots
# ─────────────────────────────────────────────
def generate_shap_plots(model, X_test: pd.DataFrame, y_proba: np.ndarray) -> None:
    """
    Generate SHAP summary, bar, and waterfall plots and save to reports/.

    Parameters
    ----------
    model : estimator
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_proba : np.ndarray
        Predicted probabilities for class 1.
    """
    print("\n🔍  Computing SHAP values (this may take a moment) …")

    # Use TreeExplainer for tree models, otherwise KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classifiers, TreeExplainer may return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (churn)
    except Exception:
        print("   Using KernelExplainer (slower, sampling 100 rows) …")
        background = shap.sample(X_test, 50)
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
        shap_values = explainer.shap_values(X_test.iloc[:100], nsamples=100)
        X_test = X_test.iloc[:100]
        y_proba = y_proba[:100]

    # 6 ── SHAP Summary (beeswarm) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title("SHAP Summary — Feature Impact on Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.REPORT_PATHS["shap_summary"], dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"✅  Saved SHAP summary plot → {config.REPORT_PATHS['shap_summary']}")

    # 7 ── SHAP Bar plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP Value| — Top 15 Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.REPORT_PATHS["shap_bar"], dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"✅  Saved SHAP bar plot → {config.REPORT_PATHS['shap_bar']}")

    # 8 ── SHAP Waterfall — 3 sample customers ────────────────────
    _save_waterfall_plot(explainer, shap_values, X_test, y_proba, "high")
    _save_waterfall_plot(explainer, shap_values, X_test, y_proba, "medium")
    _save_waterfall_plot(explainer, shap_values, X_test, y_proba, "low")


def _save_waterfall_plot(
    explainer, shap_values: np.ndarray, X_test: pd.DataFrame,
    y_proba: np.ndarray, risk_level: str,
) -> None:
    """
    Save a SHAP waterfall plot for a customer at the given risk level.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : np.ndarray
        SHAP values matrix.
    X_test : pd.DataFrame
        Test features.
    y_proba : np.ndarray
        Predicted churn probabilities.
    risk_level : str
        One of "high", "medium", "low".
    """
    if risk_level == "high":
        idx = int(np.argmax(y_proba))
        key = "shap_waterfall_high"
    elif risk_level == "medium":
        mid_idx = np.argsort(np.abs(y_proba - 0.5))
        idx = int(mid_idx[0])
        key = "shap_waterfall_medium"
    else:
        idx = int(np.argmin(y_proba))
        key = "shap_waterfall_low"

    # Build an Explanation object for this single instance
    if hasattr(explainer, "expected_value"):
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
    else:
        base_val = 0

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=base_val,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns),
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    risk_label = risk_level.capitalize()
    plt.title(f"SHAP Waterfall — {risk_label} Risk Customer (P={y_proba[idx]:.2%})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.REPORT_PATHS[key], dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"✅  Saved SHAP waterfall ({risk_label}) → {config.REPORT_PATHS[key]}")


# ─────────────────────────────────────────────
# 9. Business Summary
# ─────────────────────────────────────────────
def print_business_summary(
    y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, X_test: pd.DataFrame,
) -> None:
    """
    Print a business-friendly summary of model results.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Churn probability for each customer.
    X_test : pd.DataFrame
        Test features (must contain MonthlyCharges, possibly scaled).
    """
    total_customers = len(y_test)
    high_risk_count = int((y_proba >= config.MEDIUM_RISK_THRESHOLD).sum())
    actual_churners = int(y_test.sum())
    correctly_identified = int(((y_pred == 1) & (y_test == 1)).sum())
    recall_pct = (correctly_identified / actual_churners * 100) if actual_churners > 0 else 0

    # Estimate revenue at risk — use raw test data if possible
    try:
        raw_test = pd.read_csv(config.TEST_PATH)
        if "MonthlyCharges" in raw_test.columns:
            monthly_charges = raw_test["MonthlyCharges"].values
        else:
            monthly_charges = np.zeros(total_customers)
    except Exception:
        monthly_charges = np.zeros(total_customers)

    revenue_at_risk = float(monthly_charges[y_proba >= config.MEDIUM_RISK_THRESHOLD].sum())
    revenue_inr = revenue_at_risk * config.USD_TO_INR_RATE

    print("\n" + "=" * 60)
    print("  📊  BUSINESS SUMMARY")
    print("=" * 60)
    print(f"  Total test customers         : {total_customers:,}")
    print(f"  Flagged as HIGH churn risk    : {high_risk_count:,}")
    print(f"  Actual churners in test set   : {actual_churners:,}")
    print(f"  Correctly identified churners : {correctly_identified:,} ({recall_pct:.1f}%)")
    print(f"  Estimated monthly revenue at risk: {config.CURRENCY_SYMBOL}{revenue_inr:,.0f}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# MAIN — Run full evaluation
# ─────────────────────────────────────────────
def evaluate() -> None:
    """
    Execute the full evaluation pipeline for the best model.
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    model, X_test, y_test, feature_names, model_name = load_best_model_and_data()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, feature_names)
    generate_shap_plots(model, X_test, y_proba)
    print_business_summary(y_test, y_pred, y_proba, X_test)

    print("🎉  Model evaluation complete!\n")


if __name__ == "__main__":
    evaluate()
