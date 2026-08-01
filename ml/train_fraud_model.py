# ============================================================
#  MediSuite-AI-Agent -- ml/train_fraud_model.py
#  Feature 8: Fraud Detection Model Training
#
#  Algorithm: Isolation Forest (anomaly detection)
#  Run ONCE: python ml/train_fraud_model.py
#  Output:   models/fraud_detection_model.pkl
#            models/fraud_model_stats.json
# ============================================================

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from ml.fraud_preprocessing import FRAUD_FEATURE_COLUMNS

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "fraud_detection_dataset.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "fraud_detection_model.pkl")
STATS_PATH   = os.path.join(MODEL_DIR, "fraud_model_stats.json")

# Column mapping CSV → FRAUD_FEATURE_COLUMNS
COL_MAP = {
    "bill_amount":          "bill_amount",
    "bill_amount_zscore":   None,           # computed below
    "admission_days":       "admission_days",
    "verification_score":   "verification_score",
    "verification_status_enc": None,        # computed below
    "approval_probability": "approval_probability",
    "previous_claims":      "previous_claims",
    "claim_frequency_7d":   "claim_frequency_7d",
    "insurance_id_valid":   "insurance_id_valid",
    "policy_valid":         "policy_valid",
    "has_doctor":           "has_doctor",
    "has_disease":          "has_disease",
}

VER_ENC = {"rejected":0,"incomplete":1,"eligible":2}

def load_dataset(path):
    if not os.path.exists(path):
        print("Dataset not found. Generating...")
        from ml.generate_fraud_dataset import generate
        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate(path)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess(df):
    df = df.fillna(0)

    # Bill z-score
    mean_bill = df["bill_amount"].mean()
    std_bill  = df["bill_amount"].std()
    df["bill_amount_zscore"] = (df["bill_amount"] - mean_bill) / (std_bill if std_bill>0 else 1)

    # Verification status encoding
    if "verification_status" in df.columns:
        df["verification_status_enc"] = df["verification_status"].str.lower().map(VER_ENC).fillna(1)

    feature_cols = []
    for fc in FRAUD_FEATURE_COLUMNS:
        if fc in df.columns:
            feature_cols.append(fc)
        else:
            df[fc] = 0
            feature_cols.append(fc)

    X = df[feature_cols].values.astype(float)
    y = df["is_fraud"].values.astype(int) if "is_fraud" in df.columns else None

    stats = {"mean_bill": float(mean_bill), "std_bill": float(std_bill)}
    return X, y, stats

def train(X):
    # Train only on non-fraud data (Isolation Forest: learns normal, flags anomalies)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=0.18,  # expected fraud rate
            random_state=42,
            max_samples="auto"
        ))
    ])
    pipeline.fit(X)
    return pipeline

def evaluate(pipeline, X, y):
    # Isolation Forest: -1 = anomaly (fraud), 1 = normal
    raw_preds = pipeline.predict(X)
    pred_labels = (raw_preds == -1).astype(int)  # 1=fraud, 0=normal

    if y is not None:
        print("\n=== Fraud Detection Evaluation ===")
        print(classification_report(y, pred_labels, target_names=["Normal","Fraud"]))
        cm = confusion_matrix(y, pred_labels).tolist()
        print(f"Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        return cm
    return []

def main():
    print("MediSuite Feature 8 — Fraud Detection Model Training")
    print("="*55)

    df  = load_dataset(DATASET_PATH)
    X, y, stats = preprocess(df)
    print(f"Features: {X.shape[1]}  Samples: {X.shape[0]}")

    print("\nTraining Isolation Forest...")
    pipeline = train(X)
    print("Training complete.")

    cm = evaluate(pipeline, X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    model_stats = {
        "model_name":  "Isolation Forest",
        "n_estimators": 200,
        "contamination": 0.18,
        "mean_bill":   stats["mean_bill"],
        "std_bill":    stats["std_bill"],
        "confusion_matrix": cm
    }
    with open(STATS_PATH,"w") as f:
        json.dump(model_stats, f, indent=2)
    print(f"Stats saved: {STATS_PATH}")
    print("\nDone. Run app and visit /claim-fraud")

if __name__ == "__main__":
    main()