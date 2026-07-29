# ============================================================
#  MediSuite-AI-Agent -- ml/predict_claim.py
#  Feature 7: Prediction Module
#
#  Loads the trained model ONCE at import time.
#  Exposes predict() function for Flask and tests.
#
#  DO NOT import claim_extractor, medical_nlp, or claim_verifier here.
#  This module is fully independent.
# ============================================================

import os
import json
import joblib
from datetime import datetime

from ml.preprocessing import build_feature_vector, FEATURE_COLUMNS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "claim_prediction_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

MODEL_NAME = "Logistic Regression"

# ── Load model once at module import ─────────────────────────────────────────
_pipeline = None


def load_model():
    """
    Load the trained model pipeline from disk.
    Called once at module import — subsequent calls return cached model.
    Raises FileNotFoundError with a helpful message if model not trained yet.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Please run: python ml/train_model.py"
        )

    _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_model_metrics() -> dict:
    """Load saved training metrics for display in UI."""
    if not os.path.exists(METRICS_PATH):
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================
#  CONFIDENCE CLASSIFIER
# ============================================================

def classify_confidence(probability: float) -> str:
    """
    Map approval probability to a human-readable confidence level.
    probability: 0.0 – 100.0
    """
    if probability >= 90:
        return "High"
    elif probability >= 70:
        return "Medium"
    else:
        return "Low"


# ============================================================
#  MAIN PREDICTION FUNCTION
# ============================================================

def predict(claim_data: dict) -> dict:
    """
    Predict insurance claim approval.

    Input: claim_data dict with any of these keys:
        patient_age, gender, hospital_type, disease, bill_amount,
        admission_date, discharge_date, insurance_id, policy_number,
        verification_score, verification_status,
        previous_claims, fraud_flag

    Returns:
    {
        "prediction":           "Approved" | "Rejected",
        "approval_probability": 94.2,
        "rejection_probability": 5.8,
        "confidence":           "High" | "Medium" | "Low",
        "model_name":           "Logistic Regression",
        "prediction_time":      "2024-03-18 10:30:00",
        "feature_values":       {...}   # for debugging/display
    }

    Raises:
        FileNotFoundError — if model not trained
        ValueError — if feature vector cannot be built
    """
    # Load model (cached after first call)
    pipeline = load_model()

    # Build feature vector
    features = build_feature_vector(claim_data)

    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Feature vector length mismatch: expected {len(FEATURE_COLUMNS)}, "
            f"got {len(features)}"
        )

    import numpy as np
    X = np.array([features])

    # Predict
    pred_label   = pipeline.predict(X)[0]           # 0 = Rejected, 1 = Approved
    pred_proba   = pipeline.predict_proba(X)[0]     # [P(Rejected), P(Approved)]

    approval_prob  = round(pred_proba[1] * 100, 1)
    rejection_prob = round(pred_proba[0] * 100, 1)
    prediction     = "Approved" if pred_label == 1 else "Rejected"
    confidence     = classify_confidence(approval_prob)

    # Feature values for display (human-readable)
    feature_display = dict(zip(FEATURE_COLUMNS, features))

    return {
        "prediction":            prediction,
        "approval_probability":  approval_prob,
        "rejection_probability": rejection_prob,
        "confidence":            confidence,
        "model_name":            MODEL_NAME,
        "prediction_time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_values":        feature_display
    }


def is_model_ready() -> bool:
    """Check if the model file exists (for Flask startup check)."""
    return os.path.exists(MODEL_PATH)


# ============================================================
#  STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("MediSuite-AI-Agent — Feature 7: Prediction Test")
    print("="*55)

    if not is_model_ready():
        print("Model not found. Train it first: python ml/train_model.py")
    else:
        # Test 1: Strong approval candidate
        result = predict({
            "patient_age":        45,
            "gender":             "male",
            "hospital_type":      "private",
            "disease":            "Typhoid Fever",
            "bill_amount":        48150,
            "admission_date":     "10/03/2024",
            "discharge_date":     "17/03/2024",
            "insurance_id":       "INS/2024/001",
            "policy_number":      "POL/2024/001",
            "verification_score": 92,
            "verification_status":"Eligible",
            "previous_claims":    1,
            "fraud_flag":         0,
        })
        print(f"Test 1 — Expected: Approved")
        print(f"  Prediction   : {result['prediction']}")
        print(f"  Probability  : {result['approval_probability']}%")
        print(f"  Confidence   : {result['confidence']}")

        # Test 2: Likely rejection
        result2 = predict({
            "patient_age":        35,
            "gender":             "female",
            "hospital_type":      "private",
            "disease":            "Cancer",
            "bill_amount":        450000,
            "insurance_id":       "",       # missing
            "policy_number":      "",       # missing
            "verification_score": 15,
            "verification_status": "Rejected",
            "previous_claims":    4,
            "fraud_flag":         1,
        })
        print(f"\nTest 2 — Expected: Rejected")
        print(f"  Prediction   : {result2['prediction']}")
        print(f"  Probability  : {result2['rejection_probability']}% rejection")
        print(f"  Confidence   : {result2['confidence']}")