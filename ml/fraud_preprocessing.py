# ============================================================
#  MediSuite-AI-Agent -- ml/fraud_preprocessing.py
#  Feature 8: Fraud Detection Preprocessing
#
#  Shared between train_fraud_model.py and detect_fraud.py.
#  Converts raw claim dicts into numeric feature vectors
#  for anomaly detection models (Isolation Forest etc.)
# ============================================================

import re
import numpy as np
from datetime import datetime

# ── Feature columns (order matters — must match training) ─────────────────────
FRAUD_FEATURE_COLUMNS = [
    "bill_amount",           # raw bill amount
    "bill_amount_zscore",    # how many std devs from mean (set at inference)
    "admission_days",        # length of hospital stay
    "verification_score",    # 0–100 from Feature 6
    "verification_status_enc",  # 0=Rejected, 1=Incomplete, 2=Eligible
    "approval_probability",  # 0–100 from Feature 7 AI prediction
    "previous_claims",       # number of past claims
    "claim_frequency_7d",    # claims submitted in last 7 days
    "insurance_id_valid",    # 0/1
    "policy_valid",          # 0/1
    "has_doctor",            # 0/1
    "has_disease",           # 0/1
]

VERIFICATION_ENC = {"rejected": 0, "incomplete": 1, "eligible": 2}


def parse_float(val, default=0.0) -> float:
    try:
        return max(0.0, float(str(val).replace(",", "").strip()))
    except:
        return default


def parse_int(val, default=0) -> int:
    try:
        return max(0, int(float(str(val).strip())))
    except:
        return default


def calc_admission_days(adm_str, dis_str) -> int:
    fmts = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"]
    adm = dis = None
    for fmt in fmts:
        try: adm = datetime.strptime(str(adm_str).strip(), fmt); break
        except: pass
    for fmt in fmts:
        try: dis = datetime.strptime(str(dis_str).strip(), fmt); break
        except: pass
    if adm and dis and dis >= adm:
        return max(1, (dis - adm).days)
    return 1


def build_fraud_feature_vector(claim_data: dict,
                                mean_bill: float = 50000.0,
                                std_bill: float = 40000.0) -> list:
    """
    Build feature vector for fraud detection.

    claim_data keys:
        bill_amount, admission_date, discharge_date,
        verification_score, verification_status,
        approval_probability, previous_claims,
        claim_frequency_7d, insurance_id, policy_number,
        doctor_name, disease

    mean_bill / std_bill: population stats (from training data)
    used to compute z-score at inference time.
    """
    bill        = parse_float(claim_data.get("bill_amount", 0))
    bill_z      = (bill - mean_bill) / (std_bill if std_bill > 0 else 1)
    adm_days    = calc_admission_days(
                      claim_data.get("admission_date"),
                      claim_data.get("discharge_date"))
    ver_score   = parse_float(claim_data.get("verification_score", 50))
    ver_enc     = VERIFICATION_ENC.get(
                      str(claim_data.get("verification_status","")).lower(), 1)
    appr_prob   = parse_float(claim_data.get("approval_probability", 50))
    prev_claims = parse_int(claim_data.get("previous_claims", 0))
    freq_7d     = parse_int(claim_data.get("claim_frequency_7d", 0))
    ins_valid   = 1 if str(claim_data.get("insurance_id","")).strip() else 0
    pol_valid   = 1 if str(claim_data.get("policy_number","")).strip() else 0
    has_doctor  = 1 if str(claim_data.get("doctor_name","")).strip() else 0
    has_disease = 1 if str(claim_data.get("disease","")).strip() else 0

    return [bill, bill_z, adm_days, ver_score, ver_enc,
            appr_prob, prev_claims, freq_7d,
            ins_valid, pol_valid, has_doctor, has_disease]