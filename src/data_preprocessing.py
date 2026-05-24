"""
data_preprocessing.py
=====================
Complete data preprocessing pipeline for the Telco Customer Churn dataset.

Workflow
--------
1. Download raw data (if not present) and save to data/raw/
2. Drop customerID
3. Convert TotalCharges to numeric, fill NaN with median
4. Encode Churn → 1/0
5. Label-encode binary columns; One-Hot encode multi-category columns
6. Scale numerical columns with StandardScaler
7. Apply SMOTE to balance classes
8. Stratified train/test split (80/20)
9. Save processed CSVs to data/processed/
10. Print summary report
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")


def download_data() -> pd.DataFrame:
    """
    Download the Telco Customer Churn dataset from GitHub and save to
    data/raw/telco_churn.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded into a DataFrame.
    """
    os.makedirs(os.path.dirname(config.DATA_RAW_PATH), exist_ok=True)

    if os.path.exists(config.DATA_RAW_PATH):
        print("📂  Raw data already exists. Loading from disk …")
        df = pd.read_csv(config.DATA_RAW_PATH)
    else:
        print("⬇️   Downloading Telco Customer Churn dataset …")
        try:
            df = pd.read_csv(config.DATASET_URL)
            df.to_csv(config.DATA_RAW_PATH, index=False)
            print(f"✅  Saved raw data → {config.DATA_RAW_PATH}")
        except Exception as exc:
            raise RuntimeError(f"Failed to download dataset: {exc}") from exc

    return df


def print_summary(df: pd.DataFrame, stage: str) -> None:
    """
    Print a human-readable summary of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarise.
    stage : str
        Label such as "BEFORE PROCESSING" or "AFTER PROCESSING".
    """
    print(f"\n{'═' * 55}")
    print(f"  DATASET SUMMARY — {stage}")
    print(f"{'═' * 55}")
    print(f"  Shape           : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"  Duplicates      : {df.duplicated().sum()}")

    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts()
        churn_pct = df["Churn"].value_counts(normalize=True) * 100
        print(f"\n  Class distribution (Churn):")
        for val in sorted(churn_counts.index):
            label = "Churned" if val == 1 else ("Retained" if val == 0 else val)
            print(f"    {label:>10}: {churn_counts[val]:>5}  ({churn_pct[val]:.1f}%)")

    print(f"{'═' * 55}\n")


def preprocess() -> None:
    """
    Execute the full preprocessing pipeline and save outputs.

    Steps
    -----
    1. Load raw data.
    2. Drop customerID.
    3. Fix TotalCharges dtype.
    4. Encode Churn target.
    5. Label-encode binary columns.
    6. One-Hot encode multi-category columns.
    7. Scale numerical features.
    8. Stratified train/test split.
    9. Apply SMOTE to training set only.
    10. Save to data/processed/.
    """
    # 1 ── Load raw data ──────────────────────────────────────────
    df = download_data()
    print_summary(df, "BEFORE PROCESSING")

    # 2 ── Drop customerID ────────────────────────────────────────
    if config.ID_COLUMN in df.columns:
        df.drop(columns=[config.ID_COLUMN], inplace=True)
        print(f"🗑️   Dropped column: {config.ID_COLUMN}")

    # 3 ── TotalCharges → numeric ─────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_tc, inplace=True)
    print(f"🔢  Converted TotalCharges to numeric (median fill = {median_tc:.2f})")

    # 4 ── Encode Churn → 1/0 ─────────────────────────────────────
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print("🏷️   Encoded Churn → 1 (Yes) / 0 (No)")

    # ── Feature Engineering (inline before encoding) ─────────────
    # We do this here so raw categorical values are still available.
    df = _add_engineered_features(df)

    # 5 ── Label-encode binary columns ────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}
    binary_cols_in_data = [c for c in config.BINARY_COLUMNS if c in df.columns and c != "Churn"]
    for col in binary_cols_in_data:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print(f"🏷️   Label-encoded {len(binary_cols_in_data)} binary columns")

    # 6 ── One-Hot encode multi-category columns ──────────────────
    multi_cols_in_data = [c for c in config.MULTI_CATEGORY_COLUMNS if c in df.columns]
    # Include engineered categorical columns (tenure_group)
    if "tenure_group" in df.columns and "tenure_group" not in multi_cols_in_data:
        multi_cols_in_data.append("tenure_group")
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    ohe_array = ohe.fit_transform(df[multi_cols_in_data])
    ohe_feature_names = ohe.get_feature_names_out(multi_cols_in_data)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=df.index)
    df.drop(columns=multi_cols_in_data, inplace=True)
    df = pd.concat([df, ohe_df], axis=1)
    print(f"🔥  One-Hot encoded {len(multi_cols_in_data)} multi-category columns → {len(ohe_feature_names)} new features")

    # 7 ── Scale numerical features ───────────────────────────────
    scaler = StandardScaler()
    num_cols_in_data = [c for c in config.NUMERICAL_FEATURES if c in df.columns]
    df[num_cols_in_data] = scaler.fit_transform(df[num_cols_in_data])
    print(f"📏  Scaled numerical features: {num_cols_in_data}")

    # ── Separate features & target ───────────────────────────────
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    # 8 ── Stratified train/test split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"✂️   Train/test split → train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    # 9 ── SMOTE on training set only ─────────────────────────────
    smote = SMOTE(random_state=config.SMOTE_RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"⚖️   SMOTE applied → train now: {X_train_sm.shape[0]} rows (balanced)")

    # 10 ── Save processed data ───────────────────────────────────
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    train_df = pd.concat([X_train_sm, y_train_sm.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)
    print(f"💾  Saved train data → {config.TRAIN_PATH}")
    print(f"💾  Saved test data  → {config.TEST_PATH}")

    # ── Save scaler & encoders ───────────────────────────────────
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, config.MODEL_PATHS["scaler"])
    joblib.dump(label_encoders, config.MODEL_PATHS["label_encoders"])
    joblib.dump(ohe, config.MODEL_PATHS["ohe_encoder"])
    feature_names = list(X_train_sm.columns)
    joblib.dump(feature_names, config.MODEL_PATHS["feature_names"])
    print("💾  Saved scaler, label encoders, OHE encoder, and feature names")

    # ── Final summary ────────────────────────────────────────────
    print_summary(train_df, "AFTER PROCESSING (TRAIN)")
    print_summary(test_df, "AFTER PROCESSING (TEST)")
    print("🎉  Data preprocessing complete!\n")


# ─────────────────────────────────────────────
# INLINE FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features while raw categorical values are still available.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw (un-encoded) categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns appended.
    """
    # 1. tenure_group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=config.TENURE_BINS,
        labels=config.TENURE_LABELS,
        right=True,
    )
    print("🆕  Created feature: tenure_group")

    # 2. avg_monthly_spend
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"]).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    print("🆕  Created feature: avg_monthly_spend")

    # 3. service_count — count of "Yes" across service columns
    service_cols = [c for c in config.SERVICE_COLUMNS if c in df.columns]
    df["service_count"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if str(v).strip().lower() in ("yes", "fiber optic", "dsl")),
        axis=1,
    )
    print("🆕  Created feature: service_count")

    # 4. has_no_support
    df["has_no_support"] = (
        (df.get("TechSupport", "No").astype(str).str.strip().str.lower() == "no") &
        (df.get("OnlineSecurity", "No").astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    print("🆕  Created feature: has_no_support")

    # 5. is_high_value
    threshold_75 = df["MonthlyCharges"].quantile(0.75)
    df["is_high_value"] = (df["MonthlyCharges"] > threshold_75).astype(int)
    print(f"🆕  Created feature: is_high_value (threshold = {threshold_75:.2f})")

    # 6. payment_risk
    df["payment_risk"] = (
        df.get("PaymentMethod", "").astype(str).str.strip() == "Electronic check"
    ).astype(int)
    print("🆕  Created feature: payment_risk")

    # tenure_group was needed as an engineered feature but must be
    # encoded — add it to multi-category for OHE
    if "tenure_group" in df.columns:
        df["tenure_group"] = df["tenure_group"].astype(str)

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    preprocess()
