# ============================================================
#  MediSuite-AI-Agent -- ml/train_model.py
#  Feature 7: Model Training Script
#
#  Run ONCE to train the model:
#    python ml/train_model.py
#
#  Output:
#    models/claim_prediction_model.pkl
#    models/model_metrics.json
#
#  To retrain: just run this script again.
#  The Flask app loads the model from the .pkl file at startup.
# ============================================================

import os
import sys
import json

# ── Add project root to path so we can import ml.preprocessing ───────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)
from sklearn.pipeline import Pipeline

from ml.preprocessing import (
    FEATURE_COLUMNS, encode_gender, encode_hospital_type,
    encode_verification_status, encode_disease_severity,
    parse_amount, parse_age
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "insurance_claim_dataset.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "claim_prediction_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


# ============================================================
#  STEP 1: LOAD DATASET
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset. Generate synthetic data if missing."""
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Generating synthetic data...")
        from ml.generate_dataset import generate_dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate_dataset(path)

    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} records, {df.shape[1]} columns")
    return df


# ============================================================
#  STEP 2: PREPROCESS
# ============================================================

def preprocess_dataset(df: pd.DataFrame) -> tuple:
    """
    Convert CSV columns into model features.

    CSV columns → FEATURE_COLUMNS order:
      patient_age, gender_encoded, hospital_type_encoded,
      disease_severity, bill_amount, admission_days,
      insurance_id_valid, policy_valid, verification_score,
      verification_status_encoded, previous_claims, fraud_flag

    Returns (X: np.array, y: np.array)
    """
    # Handle missing values
    df = df.fillna({
        "patient_age": 35,
        "gender": "male",
        "hospital_type": "private",
        "disease": "",
        "bill_amount": 0,
        "admission_days": 1,
        "insurance_id_valid": 0,
        "policy_valid": 0,
        "verification_score": 50,
        "verification_status": "Incomplete",
        "previous_claims": 0,
        "fraud_flag": 0,
    })

    # Encode categorical columns
    df["gender_encoded"]              = df["gender"].apply(encode_gender)
    df["hospital_type_encoded"]       = df["hospital_type"].apply(encode_hospital_type)
    df["verification_status_encoded"] = df["verification_status"].apply(encode_verification_status)

    # Disease severity: use pre-encoded column if present, else encode
    if "disease_severity" in df.columns:
        severity_map = {"low": 0, "medium": 1, "high": 2}
        df["disease_severity_enc"] = df["disease_severity"].map(severity_map).fillna(0).astype(int)
    else:
        df["disease_severity_enc"] = df["disease"].apply(encode_disease_severity)

    # Build feature matrix using FEATURE_COLUMNS order
    col_map = {
        "patient_age":                 "patient_age",
        "gender_encoded":              "gender_encoded",
        "hospital_type_encoded":       "hospital_type_encoded",
        "disease_severity":            "disease_severity_enc",
        "bill_amount":                 "bill_amount",
        "admission_days":              "admission_days",
        "insurance_id_valid":          "insurance_id_valid",
        "policy_valid":                "policy_valid",
        "verification_score":          "verification_score",
        "verification_status_encoded": "verification_status_encoded",
        "previous_claims":             "previous_claims",
        "fraud_flag":                  "fraud_flag",
    }

    X = df[[col_map[c] for c in FEATURE_COLUMNS]].values.astype(float)
    y = df["approved"].values.astype(int)

    return X, y


# ============================================================
#  STEP 3: TRAIN MODEL
# ============================================================

def train(X_train, y_train) -> Pipeline:
    """
    Train a Logistic Regression model wrapped in a sklearn Pipeline.

    Using Pipeline ensures that scaling is applied consistently
    during both training and prediction — no data leakage.

    To swap to a different model (e.g. Random Forest), replace
    LogisticRegression() with RandomForestClassifier() here.
    The rest of the code needs no changes.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            max_iter=500,
            random_state=42,
            class_weight="balanced",  # handle imbalanced approval/rejection
            C=1.0,                    # regularization strength
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


# ============================================================
#  STEP 4: EVALUATE
# ============================================================

def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Compute and print evaluation metrics."""
    y_pred     = pipeline.predict(X_test)
    y_prob     = pipeline.predict_proba(X_test)[:, 1]

    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_prob)
    cm         = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(roc_auc, 4),
        "confusion_matrix": cm,
        "model_name": "Logistic Regression",
        "trained_on": len(y_test) + len(y_test),  # approx total
    }

    print("\n" + "="*55)
    print("MODEL EVALUATION METRICS")
    print("="*55)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected","Approved"]))

    return metrics


# ============================================================
#  STEP 5: SAVE MODEL
# ============================================================

def save_model(pipeline: Pipeline, metrics: dict):
    """Save trained model and metrics to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_PATH}")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("MediSuite-AI-Agent — Feature 7: Model Training")
    print("="*55)

    # 1. Load
    df = load_dataset(DATASET_PATH)

    # 2. Preprocess
    X, y = preprocess_dataset(df)
    print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"Approved: {y.sum()}  |  Rejected: {(y==0).sum()}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Train
    print("\nTraining Logistic Regression...")
    pipeline = train(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate
    metrics = evaluate(pipeline, X_test, y_test)

    # 6. Save
    save_model(pipeline, metrics)

    print("\nDone! The model is ready for predictions.")
    print(f"Run your Flask app and visit /claim-prediction")


if __name__ == "__main__":
    main()