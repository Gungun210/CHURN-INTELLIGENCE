"""
config.py
=========
Global configuration file for the Customer Churn Prediction & Revenue Risk Intelligence System.
All paths, constants, hyperparameter grids, and visual settings are defined here.
"""

import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_PATH       = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
DATA_PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH          = os.path.join(DATA_PROCESSED_DIR, "train.csv")
TEST_PATH           = os.path.join(DATA_PROCESSED_DIR, "test.csv")

MODELS_DIR          = os.path.join(BASE_DIR, "models")
REPORTS_DIR         = os.path.join(BASE_DIR, "reports")
NOTEBOOKS_DIR       = os.path.join(BASE_DIR, "notebooks")

# ─────────────────────────────────────────────
# DATASET URL
# ─────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
SMOTE_RANDOM_STATE = 42
CV_FOLDS     = 5

# ─────────────────────────────────────────────
# TARGET & ID COLUMNS
# ─────────────────────────────────────────────
TARGET_COLUMN = "Churn"
ID_COLUMN     = "customerID"

# ─────────────────────────────────────────────
# NUMERICAL FEATURES (to be scaled)
# ─────────────────────────────────────────────
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# ─────────────────────────────────────────────
# BINARY CATEGORICAL COLUMNS (Label Encoded)
# ─────────────────────────────────────────────
BINARY_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn"
]

# ─────────────────────────────────────────────
# MULTI-CATEGORY COLUMNS (One-Hot Encoded)
# ─────────────────────────────────────────────
MULTI_CATEGORY_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

# ─────────────────────────────────────────────
# SERVICE COLUMNS (for service_count feature)
# ─────────────────────────────────────────────
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

# ─────────────────────────────────────────────
# TENURE GROUP BINS
# ─────────────────────────────────────────────
TENURE_BINS   = [0, 12, 24, 48, 60, float("inf")]
TENURE_LABELS = ["New", "Growing", "Mature", "Loyal", "Champion"]

# ─────────────────────────────────────────────
# CHURN RISK THRESHOLDS
# ─────────────────────────────────────────────
LOW_RISK_THRESHOLD    = 0.30   # < 30% probability → Low
MEDIUM_RISK_THRESHOLD = 0.70   # 30–70% → Medium, > 70% → High

# ─────────────────────────────────────────────
# BUSINESS SETTINGS
# ─────────────────────────────────────────────
CURRENCY_SYMBOL        = "₹"
USD_TO_INR_RATE        = 83.5  # Approximate conversion rate
MONTHLY_REVENUE_COLUMN = "MonthlyCharges"

# ─────────────────────────────────────────────
# MODEL FILE PATHS
# ─────────────────────────────────────────────
MODEL_PATHS = {
    "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "random_forest":       os.path.join(MODELS_DIR, "random_forest.pkl"),
    "xgboost":             os.path.join(MODELS_DIR, "xgboost.pkl"),
    "lightgbm":            os.path.join(MODELS_DIR, "lightgbm.pkl"),
    "best_model":          os.path.join(MODELS_DIR, "best_model.pkl"),
    "scaler":              os.path.join(MODELS_DIR, "scaler.pkl"),
    "label_encoders":      os.path.join(MODELS_DIR, "label_encoders.pkl"),
    "ohe_encoder":         os.path.join(MODELS_DIR, "ohe_encoder.pkl"),
    "feature_names":       os.path.join(MODELS_DIR, "feature_names.pkl"),
    "best_model_name":     os.path.join(MODELS_DIR, "best_model_name.pkl"),
}

# ─────────────────────────────────────────────
# REPORT FILE PATHS
# ─────────────────────────────────────────────
REPORT_PATHS = {
    "confusion_matrix":       os.path.join(REPORTS_DIR, "confusion_matrix.png"),
    "roc_curve":              os.path.join(REPORTS_DIR, "roc_curve.png"),
    "pr_curve":               os.path.join(REPORTS_DIR, "precision_recall_curve.png"),
    "feature_importance":     os.path.join(REPORTS_DIR, "feature_importance.png"),
    "shap_summary":           os.path.join(REPORTS_DIR, "shap_summary.png"),
    "shap_bar":               os.path.join(REPORTS_DIR, "shap_bar.png"),
    "shap_waterfall_high":    os.path.join(REPORTS_DIR, "shap_waterfall_high_risk.png"),
    "shap_waterfall_medium":  os.path.join(REPORTS_DIR, "shap_waterfall_medium_risk.png"),
    "shap_waterfall_low":     os.path.join(REPORTS_DIR, "shap_waterfall_low_risk.png"),
    "model_comparison":       os.path.join(REPORTS_DIR, "model_comparison.csv"),
}

# ─────────────────────────────────────────────
# HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [500],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

LIGHTGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "subsample": [0.8, 1.0],
}

# ─────────────────────────────────────────────
# DASHBOARD COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    "primary":       "#1f4e79",   # Deep blue
    "accent":        "#e05c5c",   # Coral red
    "secondary":     "#2e86ab",   # Sky blue
    "success":       "#27ae60",   # Green
    "warning":       "#f39c12",   # Amber
    "danger":        "#e74c3c",   # Red
    "light":         "#ecf0f1",   # Light grey
    "dark":          "#2c3e50",   # Dark navy
    "background":    "#0d1b2a",   # Dashboard background
    "card":          "#1a2a3a",   # Card background
    "text":          "#ffffff",   # White text
    "muted":         "#95a5a6",   # Muted text
    "gradient_start":"#1f4e79",
    "gradient_end":  "#2e86ab",
    "churn_yes":     "#e05c5c",
    "churn_no":      "#27ae60",
}

# Plotly color sequence for consistent charts
PLOTLY_COLOR_SEQUENCE = [
    "#1f4e79", "#e05c5c", "#2e86ab", "#f39c12",
    "#27ae60", "#9b59b6", "#1abc9c", "#e67e22"
]
