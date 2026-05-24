"""
predict.py
==========
Reusable prediction pipeline for the Customer Churn Prediction System.

Given a single customer's raw data as a dictionary, this module will:
1. Apply all preprocessing and feature engineering automatically.
2. Return churn probability (0–100 %).
3. Return risk category: Low / Medium / High.
4. Return top-3 reasons via SHAP in plain English.
5. Return a recommended business action.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import shap
import joblib

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# HUMAN-READABLE FEATURE NAME MAP
# ─────────────────────────────────────────────
_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "tenure":                 "Account tenure (months)",
    "MonthlyCharges":         "Monthly bill amount",
    "TotalCharges":           "Total amount charged",
    "avg_monthly_spend":      "Average monthly spend",
    "service_count":          "Number of services used",
    "has_no_support":         "No tech support or security",
    "is_high_value":          "High-value customer flag",
    "payment_risk":           "Electronic check payment",
    "Contract_One year":      "One-year contract",
    "Contract_Two year":      "Two-year contract",
    "InternetService_Fiber optic": "Fiber optic internet",
    "InternetService_No":     "No internet service",
    "OnlineSecurity_Yes":     "Online security add-on",
    "TechSupport_Yes":        "Tech support add-on",
    "PaperlessBilling":       "Paperless billing",
    "Dependents":             "Has dependents",
    "Partner":                "Has a partner",
    "gender":                 "Customer gender",
    "PhoneService":           "Phone service",
    "StreamingTV_Yes":        "Streaming TV add-on",
    "StreamingMovies_Yes":    "Streaming movies add-on",
    "OnlineBackup_Yes":       "Online backup add-on",
    "DeviceProtection_Yes":   "Device protection add-on",
    "MultipleLines_Yes":      "Multiple phone lines",
    "tenure_group_Growing":   "Growing tenure (13-24 mo)",
    "tenure_group_Mature":    "Mature tenure (25-48 mo)",
    "tenure_group_Loyal":     "Loyal tenure (49-60 mo)",
    "tenure_group_Champion":  "Champion tenure (61+ mo)",
}


def _get_display_name(feature: str) -> str:
    """
    Return a human-readable name for a model feature.

    Parameters
    ----------
    feature : str
        Raw feature name from the model.

    Returns
    -------
    str
        Human-friendly name.
    """
    return _FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def _shap_direction(shap_val: float, feature: str, raw_val) -> str:
    """
    Generate a plain-English explanation of a SHAP value's direction.

    Parameters
    ----------
    shap_val : float
        SHAP value for this feature.
    feature : str
        Feature name.
    raw_val : any
        Raw feature value.

    Returns
    -------
    str
        Explanation sentence.
    """
    display = _get_display_name(feature)
    direction = "increases" if shap_val > 0 else "decreases"
    return f"{display} {direction} churn risk"


class ChurnPredictor:
    """
    End-to-end churn prediction for a single customer.

    Attributes
    ----------
    model : estimator
        Best trained model loaded from disk.
    scaler : StandardScaler
        Fitted scaler for numerical features.
    label_encoders : dict
        Fitted LabelEncoders for binary categorical columns.
    ohe : OneHotEncoder
        Fitted OneHotEncoder for multi-category columns.
    feature_names : list
        Ordered feature names expected by the model.
    explainer : shap.Explainer
        SHAP explainer for the loaded model.
    """

    def __init__(self) -> None:
        """Load all persisted artefacts from the models/ directory."""
        try:
            self.model = joblib.load(config.MODEL_PATHS["best_model"])
            self.scaler = joblib.load(config.MODEL_PATHS["scaler"])
            self.label_encoders = joblib.load(config.MODEL_PATHS["label_encoders"])
            self.ohe = joblib.load(config.MODEL_PATHS["ohe_encoder"])
            self.feature_names = joblib.load(config.MODEL_PATHS["feature_names"])
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Model artefacts not found. Run train_model.py first."
            ) from exc

        # Build SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            background = pd.DataFrame(
                np.zeros((1, len(self.feature_names))),
                columns=self.feature_names,
            )
            self.explainer = shap.KernelExplainer(lambda x: self.model.predict_proba(x)[:, 1], background)

    # ─────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────
    def _preprocess(self, customer: dict) -> pd.DataFrame:
        """
        Transform a raw customer dictionary into a model-ready DataFrame.

        Parameters
        ----------
        customer : dict
            Raw customer data with original column names and values.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame aligned to self.feature_names.
        """
        df = pd.DataFrame([customer])

        # ── Drop customerID if present ────────────────────────────
        if config.ID_COLUMN in df.columns:
            df.drop(columns=[config.ID_COLUMN], inplace=True)

        # ── TotalCharges → numeric ────────────────────────────────
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        # ── Feature engineering (on raw values) ───────────────────
        df = self._add_features(df)

        # ── Label-encode binary columns ───────────────────────────
        for col, le in self.label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    df[col] = 0  # safe fallback

        # ── One-Hot encode multi-category columns ─────────────────
        multi_cols = [c for c in config.MULTI_CATEGORY_COLUMNS if c in df.columns]
        if "tenure_group" in df.columns and "tenure_group" not in multi_cols:
            multi_cols.append("tenure_group")
        if multi_cols:
            ohe_array = self.ohe.transform(df[multi_cols])
            ohe_names = self.ohe.get_feature_names_out(multi_cols)
            ohe_df = pd.DataFrame(ohe_array, columns=ohe_names, index=df.index)
            df.drop(columns=multi_cols, inplace=True)
            df = pd.concat([df, ohe_df], axis=1)

        # ── Scale numerical features ─────────────────────────────
        num_cols = [c for c in config.NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        # ── Align to model feature order ─────────────────────────
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        return df

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to a single-row DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw single-row customer data.

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features added.
        """
        # tenure_group
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=config.TENURE_BINS, labels=config.TENURE_LABELS, right=True,
        ).astype(str)

        # avg_monthly_spend
        tenure_val = df["tenure"].iloc[0]
        total_val = df["TotalCharges"].iloc[0]
        df["avg_monthly_spend"] = total_val / tenure_val if tenure_val > 0 else 0

        # service_count
        svc_cols = [c for c in config.SERVICE_COLUMNS if c in df.columns]
        df["service_count"] = df[svc_cols].apply(
            lambda row: sum(
                1 for v in row if str(v).strip().lower() in ("yes", "fiber optic", "dsl")
            ), axis=1,
        )

        # has_no_support
        tech = str(df.get("TechSupport", pd.Series(["No"])).iloc[0]).strip().lower()
        sec = str(df.get("OnlineSecurity", pd.Series(["No"])).iloc[0]).strip().lower()
        df["has_no_support"] = int(tech == "no" and sec == "no")

        # is_high_value  (use training-data 75th pct ≈ 89.85)
        df["is_high_value"] = int(df["MonthlyCharges"].iloc[0] > 89.85)

        # payment_risk
        pm = str(df.get("PaymentMethod", pd.Series([""])).iloc[0]).strip()
        df["payment_risk"] = int(pm == "Electronic check")

        return df

    # ─────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────
    def predict(self, customer: dict) -> dict:
        """
        Predict churn risk for a single customer.

        Parameters
        ----------
        customer : dict
            Raw customer data with original column names/values.
            Example keys: tenure, MonthlyCharges, Contract, InternetService, etc.

        Returns
        -------
        dict
            {
                "churn_probability": float (0–100),
                "risk_category": str,
                "top_reasons": list[str],
                "recommended_action": str,
                "shap_values": np.ndarray,   # for optional visualisation
                "feature_names": list[str],
            }
        """
        X = self._preprocess(customer)

        # Probability
        proba = self.model.predict_proba(X)[0][1]
        churn_pct = round(proba * 100, 2)

        # Risk category
        if proba < config.LOW_RISK_THRESHOLD:
            risk = "Low"
            action = "No action needed"
        elif proba < config.MEDIUM_RISK_THRESHOLD:
            risk = "Medium"
            action = "Monitor closely, send engagement campaign"
        else:
            risk = "High"
            action = "Immediate retention offer recommended"

        # SHAP top reasons
        try:
            sv = self.explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]
        except Exception:
            sv = np.zeros(len(self.feature_names))

        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:3]
        reasons = [
            _shap_direction(sv[i], self.feature_names[i], X.iloc[0, i])
            for i in top_idx
        ]

        return {
            "churn_probability": churn_pct,
            "risk_category": risk,
            "top_reasons": reasons,
            "recommended_action": action,
            "shap_values": sv,
            "feature_names": self.feature_names,
        }


# ─────────────────────────────────────────────
# MAIN — Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 151.0,
    }

    print("=" * 55)
    print("  CHURN PREDICTION — Sample Customer Test")
    print("=" * 55)

    predictor = ChurnPredictor()
    result = predictor.predict(sample_customer)

    print(f"\n  Churn Probability : {result['churn_probability']:.1f}%")
    print(f"  Risk Category     : {result['risk_category']}")
    print(f"  Recommended Action: {result['recommended_action']}")
    print(f"\n  Top 3 Reasons:")
    for i, reason in enumerate(result["top_reasons"], 1):
        print(f"    {i}. {reason}")

    print("\n🎉  Prediction pipeline test complete!\n")
