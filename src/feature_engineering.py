"""
feature_engineering.py
======================
Feature engineering module for the Customer Churn Prediction System.

This module provides standalone functions to create engineered features.
The actual feature creation happens inside data_preprocessing.py (inline,
before encoding), but this module is kept as a reusable library and can
be imported independently for ad-hoc analysis or the prediction pipeline.

Engineered Features
-------------------
1. tenure_group      – categorical bin of tenure
2. avg_monthly_spend – TotalCharges / tenure
3. service_count     – number of subscribed services
4. has_no_support    – flag: no TechSupport AND no OnlineSecurity
5. is_high_value     – flag: MonthlyCharges > 75th percentile
6. payment_risk      – flag: PaymentMethod == "Electronic check"
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure into categorical groups.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'tenure' column (numeric).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'tenure_group' column.
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=config.TENURE_BINS,
        labels=config.TENURE_LABELS,
        right=True,
    )
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly spend = TotalCharges / tenure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'TotalCharges' and 'tenure' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'avg_monthly_spend' column.
    """
    df = df.copy()
    df["avg_monthly_spend"] = (
        df["TotalCharges"] / df["tenure"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of services a customer subscribes to.

    A service is counted if its column value is "Yes", "Fiber optic", or "DSL".

    Parameters
    ----------
    df : pd.DataFrame
        Must contain service columns listed in config.SERVICE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'service_count' column.
    """
    df = df.copy()
    service_cols = [c for c in config.SERVICE_COLUMNS if c in df.columns]
    df["service_count"] = df[service_cols].apply(
        lambda row: sum(
            1 for v in row
            if str(v).strip().lower() in ("yes", "fiber optic", "dsl")
        ),
        axis=1,
    )
    return df


def add_has_no_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers who lack BOTH TechSupport and OnlineSecurity.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'TechSupport' and 'OnlineSecurity' columns
        with raw categorical values.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'has_no_support' column (1/0).
    """
    df = df.copy()
    no_tech = df.get("TechSupport", pd.Series(["No"] * len(df))).astype(str).str.strip().str.lower() == "no"
    no_sec = df.get("OnlineSecurity", pd.Series(["No"] * len(df))).astype(str).str.strip().str.lower() == "no"
    df["has_no_support"] = (no_tech & no_sec).astype(int)
    return df


def add_is_high_value(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """
    Flag customers whose MonthlyCharges exceed the 75th percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'MonthlyCharges' column.
    threshold : float, optional
        Custom threshold; defaults to the 75th percentile of the column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'is_high_value' column (1/0).
    """
    df = df.copy()
    if threshold is None:
        threshold = df["MonthlyCharges"].quantile(0.75)
    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)
    return df


def add_payment_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers paying via "Electronic check" (highest historical churn rate).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'PaymentMethod' column with raw values.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'payment_risk' column (1/0).
    """
    df = df.copy()
    df["payment_risk"] = (
        df.get("PaymentMethod", pd.Series([""] * len(df)))
        .astype(str)
        .str.strip() == "Electronic check"
    ).astype(int)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ALL feature engineering steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or semi-processed DataFrame with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with all 6 engineered features added.
    """
    df = add_tenure_group(df)
    df = add_avg_monthly_spend(df)
    df = add_service_count(df)
    df = add_has_no_support(df)
    df = add_is_high_value(df)
    df = add_payment_risk(df)
    return df


# ─────────────────────────────────────────────
# MAIN — Verification
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FEATURE ENGINEERING — Verification Run")
    print("=" * 55)

    try:
        raw_df = pd.read_csv(config.DATA_RAW_PATH)
    except FileNotFoundError:
        print("⚠️  Raw data not found. Run data_preprocessing.py first to download it.")
        sys.exit(1)

    # Convert TotalCharges to numeric for feature engineering
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    raw_df["TotalCharges"].fillna(raw_df["TotalCharges"].median(), inplace=True)

    original_cols = set(raw_df.columns)
    enriched = engineer_all_features(raw_df)
    new_cols = set(enriched.columns) - original_cols

    print(f"\n✅  Engineered {len(new_cols)} new features:")
    for col in sorted(new_cols):
        n_unique = enriched[col].nunique()
        print(f"    • {col:25s}  unique values: {n_unique}")

    print(f"\n📊  DataFrame shape: {enriched.shape}")
    print(enriched[list(new_cols)].head(10).to_string(index=False))
    print("\n🎉  Feature engineering verification complete!\n")
