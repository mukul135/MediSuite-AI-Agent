# ============================================================
#  MediSuite-AI-Agent -- ml/detect_fraud.py
#  Feature 8: Fraud Detection Engine
#
#  Hybrid: Rule-Based + Isolation Forest anomaly detection.
#  Independent of all previous modules.
#  DO NOT import claim_extractor, medical_nlp, claim_verifier,
#  or predict_claim from here.
# ============================================================

import os, json
import numpy as np
import joblib
from datetime import datetime

from ml.fraud_preprocessing import build_fraud_feature_vector, parse_float, parse_int

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl")
STATS_PATH  = os.path.join(BASE_DIR, "models", "fraud_model_stats.json")

MODEL_NAME  = "Isolation Forest"

# ── Thresholds (configurable) ─────────────────────────────────────────────────
HIGH_BILL_MULTIPLIER  = 2.5   # >2.5x avg = suspicious
MAX_CLAIMS_7D         = 3     # >3 claims in 7 days = suspicious
MAX_TOTAL_CLAIMS      = 6     # >6 lifetime claims = review
BILL_ABS_HIGH         = 300_000  # Rs 3 lakh hard threshold

# ── Load model once ───────────────────────────────────────────────────────────
_pipeline   = None
_model_stats = {}

def _load_model():
    global _pipeline, _model_stats
    if _pipeline is not None:
        return _pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Fraud model not found. Run: python ml/train_fraud_model.py")
    _pipeline = joblib.load(MODEL_PATH)
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            _model_stats = json.load(f)
    return _pipeline


def is_fraud_model_ready() -> bool:
    return os.path.exists(MODEL_PATH)


def get_fraud_model_stats() -> dict:
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH) as f:
                return json.load(f)
        except:
            pass
    return {}


# ============================================================
#  SECTION 1: RULE-BASED CHECKS
#  Each returns (triggered: bool, reason: str, severity: int)
#  severity: 10=low, 25=medium, 40=high, 60=fraud
# ============================================================

def rule_no_insurance_id(data: dict):
    if not str(data.get("insurance_id","")).strip():
        return True, "Insurance ID Missing", 40
    return False, "", 0

def rule_no_policy(data: dict):
    if not str(data.get("policy_number","")).strip():
        return True, "Policy Number Missing", 25
    return False, "", 0

def rule_high_bill(data: dict, mean_bill: float):
    bill = parse_float(data.get("bill_amount",0))
    if bill > BILL_ABS_HIGH:
        pct = int((bill/mean_bill - 1)*100) if mean_bill > 0 else 0
        return True, f"Bill Amount Rs.{bill:,.0f} exceeds Rs.{BILL_ABS_HIGH:,} (avg +{pct}%)", 40
    if mean_bill > 0 and bill > mean_bill * HIGH_BILL_MULTIPLIER:
        pct = int((bill/mean_bill - 1)*100)
        return True, f"Bill Rs.{bill:,.0f} is {pct}% above average", 25
    return False, "", 0

def rule_claim_frequency(data: dict):
    freq = parse_int(data.get("claim_frequency_7d",0))
    if freq > MAX_CLAIMS_7D:
        return True, f"{freq} claims submitted in last 7 days (max {MAX_CLAIMS_7D})", 40
    prev = parse_int(data.get("previous_claims",0))
    if prev > MAX_TOTAL_CLAIMS:
        return True, f"{prev} previous claims (excessive)", 25
    return False, "", 0

def rule_invalid_dates(data: dict):
    from datetime import datetime
    adm_str = data.get("admission_date","")
    dis_str = data.get("discharge_date","")
    if not adm_str or not dis_str:
        return False, "", 0
    fmts = ["%d/%m/%Y","%d-%m-%Y","%Y-%m-%d","%d %b %Y"]
    adm = dis = None
    for fmt in fmts:
        try: adm = datetime.strptime(str(adm_str).strip(), fmt); break
        except: pass
    for fmt in fmts:
        try: dis = datetime.strptime(str(dis_str).strip(), fmt); break
        except: pass
    if adm and dis and adm > dis:
        return True, f"Admission date ({adm_str}) is after discharge date ({dis_str})", 60
    return False, "", 0

def rule_verification_rejected(data: dict):
    ver = str(data.get("verification_status","")).lower()
    appr = parse_float(data.get("approval_probability",50))
    if ver == "rejected" and appr > 70:
        return True, "Claim rejected in verification but AI predicts approval — inconsistency", 40
    if ver == "rejected":
        return True, "Verification status is Rejected", 25
    return False, "", 0

def rule_low_ver_score(data: dict):
    score = parse_float(data.get("verification_score",100))
    if score < 30:
        return True, f"Very low verification score ({score:.0f}/100)", 25
    return False, "", 0

def rule_missing_doctor(data: dict):
    if not str(data.get("doctor_name","")).strip():
        return True, "Doctor Name not provided", 10
    return False, "", 0

def rule_missing_disease(data: dict):
    if not str(data.get("disease","")).strip():
        return True, "Disease / Diagnosis not specified", 10
    return False, "", 0

def rule_duplicate_claim(data: dict):
    """Check if insurance_id + bill_amount combo appears in past_claims list."""
    past = data.get("past_claims_summary", [])
    cur_ins = str(data.get("insurance_id","")).strip()
    cur_bill = parse_float(data.get("bill_amount",0))
    for p in past:
        if (str(p.get("insurance_id","")).strip() == cur_ins
                and abs(parse_float(p.get("bill_amount",0)) - cur_bill) < 1):
            return True, f"Duplicate claim: same Insurance ID and Bill Amount as claim #{p.get('id','?')}", 60
    return False, "", 0

ALL_RULES = [
    rule_no_insurance_id,
    rule_no_policy,
    rule_high_bill,
    rule_claim_frequency,
    rule_invalid_dates,
    rule_verification_rejected,
    rule_low_ver_score,
    rule_missing_doctor,
    rule_missing_disease,
    rule_duplicate_claim,
]


# ============================================================
#  SECTION 2: SCORE → STATUS
# ============================================================

def score_to_status(score: int) -> str:
    if score >= 80: return "Fraud Suspected"
    if score >= 60: return "High Risk"
    if score >= 30: return "Medium Risk"
    return "Low Risk"


def score_to_recommendation(status: str) -> str:
    return {
        "Fraud Suspected": "Immediately flag for investigation. Do NOT approve.",
        "High Risk":       "Manual review required before approval.",
        "Medium Risk":     "Additional verification recommended.",
        "Low Risk":        "No suspicious activity detected. Safe to proceed.",
    }.get(status, "Review recommended.")


# ============================================================
#  SECTION 3: ML ANOMALY DETECTION
# ============================================================

def ml_anomaly_score(data: dict, mean_bill: float, std_bill: float) -> tuple:
    """
    Returns (anomaly_detected: bool, fraud_probability: float)
    Isolation Forest: decision_function gives anomaly score.
    More negative = more anomalous.
    We convert to 0–100 fraud probability.
    """
    try:
        pipeline = _load_model()
        fv = build_fraud_feature_vector(data, mean_bill, std_bill)
        X  = np.array([fv])

        # decision_function: lower = more anomalous (negative = outlier)
        score = pipeline.decision_function(X)[0]
        pred  = pipeline.predict(X)[0]   # -1=anomaly, 1=normal

        anomaly_detected = (pred == -1)

        # Convert score to 0–100 probability (invert: more negative = higher fraud prob)
        # Typical range is roughly -0.3 to 0.3
        clipped = max(-0.5, min(0.5, score))
        fraud_prob = round((0.5 - clipped) * 100, 1)  # 0 when normal, 100 when very anomalous

        return anomaly_detected, fraud_prob

    except FileNotFoundError:
        # Model not trained — return neutral
        return False, 50.0
    except Exception:
        return False, 50.0


# ============================================================
#  SECTION 4: MAIN DETECTION PIPELINE
# ============================================================

def detect_fraud(claim_data: dict) -> dict:
    """
    Main entry point for fraud detection.

    claim_data keys (all optional with defaults):
        bill_amount, admission_date, discharge_date,
        insurance_id, policy_number, doctor_name, disease,
        verification_score, verification_status,
        approval_probability, previous_claims,
        claim_frequency_7d, past_claims_summary (list of dicts)

    Returns structured fraud detection result dict.
    """
    stats     = get_fraud_model_stats()
    mean_bill = float(stats.get("mean_bill", 50000))
    std_bill  = float(stats.get("std_bill", 40000))

    # ── Step 1: Run all rules ─────────────────────────────────────────────────
    triggered_rules = []
    rule_score = 0

    for rule_fn in ALL_RULES:
        try:
            if rule_fn.__name__ == "rule_high_bill":
                triggered, reason, sev = rule_fn(claim_data, mean_bill)
            else:
                triggered, reason, sev = rule_fn(claim_data)
        except Exception as e:
            continue

        if triggered:
            triggered_rules.append(reason)
            rule_score = min(100, rule_score + sev)

    # ── Step 2: ML anomaly detection ─────────────────────────────────────────
    anomaly_detected, fraud_prob_ml = ml_anomaly_score(claim_data, mean_bill, std_bill)

    # ── Step 3: Combine scores (70% rules, 30% ML) ───────────────────────────
    fraud_score = int(rule_score * 0.70 + fraud_prob_ml * 0.30)
    fraud_score = max(0, min(100, fraud_score))

    # ── Step 4: Final fraud probability (blend) ───────────────────────────────
    fraud_probability = round(
        (fraud_score * 0.70 + fraud_prob_ml * 0.30), 1
    )

    # ── Step 5: Status + recommendation ──────────────────────────────────────
    fraud_status      = score_to_status(fraud_score)
    recommendation    = score_to_recommendation(fraud_status)

    return {
        "fraud_status":       fraud_status,
        "fraud_score":        fraud_score,
        "fraud_probability":  fraud_probability,
        "detected_rules":     triggered_rules,
        "anomaly_detected":   anomaly_detected,
        "recommendation":     recommendation,
        "model_name":         MODEL_NAME,
        "rule_score":         rule_score,
        "ml_score":           round(fraud_prob_ml, 1),
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error":              None
    }


# ============================================================
#  STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("="*55)
    print("TEST 1: Low Risk")
    r = detect_fraud({
        "bill_amount": 48000, "insurance_id": "INS/001",
        "policy_number": "POL/001", "doctor_name": "Dr. Priya",
        "disease": "Typhoid", "verification_score": 92,
        "verification_status": "Eligible", "approval_probability": 94,
        "previous_claims": 1, "claim_frequency_7d": 0,
        "admission_date": "10/03/2024", "discharge_date": "17/03/2024"
    })
    print(f"Status: {r['fraud_status']}  Score: {r['fraud_score']}")

    print("\nTEST 2: Fraud Suspected")
    r2 = detect_fraud({
        "bill_amount": 450000, "insurance_id": "",
        "policy_number": "", "verification_score": 10,
        "verification_status": "Rejected", "approval_probability": 85,
        "previous_claims": 8, "claim_frequency_7d": 5,
        "admission_date": "20/03/2024", "discharge_date": "10/03/2024",
    })
    print(f"Status: {r2['fraud_status']}  Score: {r2['fraud_score']}")
    print(f"Rules:  {r2['detected_rules']}")