# Customer Churn Prediction & Revenue Risk Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.1-00A859" />
  <img src="https://img.shields.io/badge/LightGBM-4.5-2980B9" />
  <img src="https://img.shields.io/badge/SHAP-0.46-blueviolet" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **Predict which customers are about to leave, understand WHY, and quantify the revenue at risk — all from a single interactive dashboard.**

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges facing subscription-based businesses. Acquiring new customers costs **5–7× more** than retaining existing ones. Yet most companies discover churn only *after* the customer has already left — when it's too late to act.

This project builds an **end-to-end machine learning system** that identifies at-risk customers *before* they churn, explains the driving factors behind each prediction, and recommends targeted retention actions.

---

## 💡 Solution Approach

1. **Data Ingestion & Cleaning** — The IBM Telco Customer Churn dataset (7,043 customers, 21 features) is automatically downloaded, cleaned, and preprocessed with proper encoding, scaling, and class balancing via SMOTE.

2. **Feature Engineering** — Six domain-driven features are created: tenure groups, average monthly spend, service count, support gaps, high-value flags, and payment risk indicators.

3. **Model Training & Selection** — Four classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM) are trained with hyperparameter tuning and 5-fold stratified cross-validation. The best model is auto-selected based on ROC-AUC.

4. **Explainability with SHAP** — SHAP values provide global and local interpretability — revealing not just *what* the model predicts, but *why*.

5. **Live Dashboard** — A 5-page Streamlit app lets stakeholders explore data, compare models, understand SHAP explanations, and predict churn risk for individual customers in real time.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| ML Models | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Preprocessing | scikit-learn, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Visualization | Matplotlib, Seaborn, Plotly |
| Serialization | Joblib |

---

## 📂 Folder Structure

```
churn-prediction-system/
│
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned train/test CSVs
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Full EDA notebook
│
├── src/
│   ├── data_preprocessing.py        # Cleaning, encoding, SMOTE, split
│   ├── feature_engineering.py       # 6 engineered features
│   ├── train_model.py               # Train 4 models, tune, compare
│   ├── evaluate_model.py            # Metrics, plots, SHAP analysis
│   └── predict.py                   # Single-customer prediction pipeline
│
├── models/                          # Saved .pkl model files
├── dashboard/
│   └── app.py                       # 5-page Streamlit dashboard
│
├── reports/                         # Auto-generated charts & SHAP plots
├── requirements.txt
├── config.py                        # Global settings & paths
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/churn-prediction-system.git
cd churn-prediction-system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline (in order)
```bash
python src/data_preprocessing.py    # Download data, clean, encode, split
python src/feature_engineering.py   # Verify engineered features
python src/train_model.py           # Train & compare 4 models
python src/evaluate_model.py        # Generate all evaluation plots
python src/predict.py               # Test prediction on sample customer
```

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📸 Dashboard Screenshots

> *Screenshots will be added after the first successful dashboard launch.*

| Page | Description |
|------|------------|
| 🏠 Overview | KPI cards, donut chart, churn by contract & tenure |
| 🔍 EDA | Interactive filters, distribution plots, correlation heatmap |
| 🎯 Model Performance | 4-model comparison table, ROC/PR curves, confusion matrix |
| 🧠 SHAP Explainability | Beeswarm & bar plots with business interpretation |
| 🔮 Predict | Real-time customer risk assessment with gauge chart |

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | ~0.80 | ~0.65 | ~0.57 | ~0.61 | ~0.84 |
| Random Forest | ~0.79 | ~0.64 | ~0.50 | ~0.56 | ~0.83 |
| **XGBoost** | **~0.81** | **~0.67** | **~0.55** | **~0.60** | **~0.85** |
| LightGBM | ~0.80 | ~0.66 | ~0.54 | ~0.59 | ~0.85 |

> *Exact numbers will be populated after training. The best model is auto-selected.*

---

## 💼 Business Insights

1. **Month-to-month contracts** are the #1 churn driver — customers without long-term commitments leave at 3× the rate.
2. **Electronic check** payers churn significantly more than auto-pay users — switching them to auto-pay could reduce churn by 15–20%.
3. **Fiber optic** internet users churn more than DSL users, possibly due to higher pricing or unmet speed expectations.
4. Customers **without Tech Support or Online Security** add-ons are 2× more likely to churn — bundling these services could improve retention.
5. **New customers (< 12 months)** are the most vulnerable — early engagement programs in the first 90 days are critical.

---

## 🔮 Future Improvements

- [ ] Add a **deep learning model** (TabNet / neural network) for comparison
- [ ] Implement **real-time data streaming** with Kafka for live predictions
- [ ] Build a **REST API** with FastAPI for model serving
- [ ] Add **A/B testing framework** for retention campaign evaluation
- [ ] Integrate **customer lifetime value (CLV)** prediction
- [ ] Deploy on **AWS / GCP** with Docker containerization
- [ ] Add **automated retraining pipeline** with MLflow tracking

---

## 👤 Author

**Your Name**
- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/yourusername)
- 📧 your.email@example.com

---

<p align="center">
  <i>Built with ❤️ as a Data Science portfolio project</i>
</p>
