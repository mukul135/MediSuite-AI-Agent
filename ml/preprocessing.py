# ============================================================
#  MediSuite-AI-Agent -- ml/preprocessing.py
#  Feature 7: Shared preprocessing logic
#
#  Used by BOTH train_model.py and predict_claim.py
#  Keeps preprocessing consistent between training and inference.
#  Never modify the input data in-place.
# ============================================================

import re
import numpy as np

# ── Feature columns the model was trained on (order matters!) ────────────────
# If you add/remove features, retrain the model.
FEATURE_COLUMNS = [
    "patient_age",
    "gender_encoded",          # 0=Female, 1=Male, 2=Other
    "hospital_type_encoded",   # 0=Government, 1=Private, 2=Trust
    "disease_severity",        # 0=Low, 1=Medium, 2=High
    "bill_amount",
    "admission_days",
    "insurance_id_valid",      # 0=No, 1=Yes
    "policy_valid",            # 0=No, 1=Yes
    "verification_score",      # 0–100
    "verification_status_encoded",  # 0=Rejected, 1=Incomplete, 2=Eligible
    "previous_claims",
    "fraud_flag",              # 0=No, 1=Yes (future use, default 0)
]


# ── Encodings ─────────────────────────────────────────────────────────────────

GENDER_MAP = {
    "male": 1, "m": 1,
    "female": 0, "f": 0,
    "other": 2
}

HOSPITAL_TYPE_MAP = {
    "government": 0, "govt": 0, "public": 0,
    "private": 1,
    "trust": 2, "ngo": 2, "charitable": 2
}

VERIFICATION_STATUS_MAP = {
    "rejected": 0,
    "incomplete": 1,
    "eligible": 2,
    "": 1  # unknown → treat as incomplete
}

# Disease severity mapping — extend as needed
HIGH_SEVERITY_DISEASES = {
    "cancer", "carcinoma", "tumor", "cardiac arrest", "heart failure",
    "sepsis", "stroke", "icu", "kidney failure", "renal failure",
    "liver failure", "leukemia", "lymphoma", "pulmonary embolism"
}

MEDIUM_SEVERITY_DISEASES = {
    "diabetes", "hypertension", "pneumonia", "tuberculosis", "tb",
    "dengue", "typhoid", "appendicitis", "fracture", "hernia",
    "asthma", "copd", "hepatitis", "pancreatitis", "epilepsy"
}

# Everything else = low severity


def encode_gender(gender_str: str) -> int:
    """Encode gender string to integer."""
    if not gender_str:
        return 1  # default Male
    return GENDER_MAP.get(str(gender_str).lower().strip(), 2)


def encode_hospital_type(hospital_type_str: str) -> int:
    """Encode hospital type to integer."""
    if not hospital_type_str:
        return 1  # default Private
    return HOSPITAL_TYPE_MAP.get(str(hospital_type_str).lower().strip(), 1)


def encode_verification_status(status_str: str) -> int:
    """Encode verification status to integer."""
    if not status_str:
        return 1
    return VERIFICATION_STATUS_MAP.get(str(status_str).lower().strip(), 1)


def encode_disease_severity(disease_str: str) -> int:
    """
    Encode disease name to severity level.
    0 = Low, 1 = Medium, 2 = High
    """
    if not disease_str:
        return 0
    disease_lower = str(disease_str).lower()
    for d in HIGH_SEVERITY_DISEASES:
        if d in disease_lower:
            return 2
    for d in MEDIUM_SEVERITY_DISEASES:
        if d in disease_lower:
            return 1
    return 0


def parse_amount(amount_val) -> float:
    """Parse bill amount from string or number."""
    if amount_val is None:
        return 0.0
    try:
        cleaned = str(amount_val).replace(",", "").strip()
        return max(0.0, float(cleaned))
    except (ValueError, TypeError):
        return 0.0


def parse_age(age_val) -> int:
    """Parse patient age from string or int."""
    if age_val is None:
        return 35  # default
    try:
        return max(0, min(120, int(str(age_val).strip())))
    except (ValueError, TypeError):
        return 35


def calculate_admission_days(admission_date_str: str, discharge_date_str: str) -> int:
    """
    Calculate number of admission days from date strings.
    Returns 0 if dates are missing or unparseable.
    """
    from datetime import datetime

    if not admission_date_str or not discharge_date_str:
        return 1  # default 1 day

    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d %b %Y", "%d %B %Y"
    ]

    adm = dis = None
    for fmt in formats:
        try:
            adm = datetime.strptime(str(admission_date_str).strip(), fmt)
            break
        except ValueError:
            continue
    for fmt in formats:
        try:
            dis = datetime.strptime(str(discharge_date_str).strip(), fmt)
            break
        except ValueError:
            continue

    if adm and dis and dis >= adm:
        return max(1, (dis - adm).days)
    return 1


def build_feature_vector(claim_data: dict) -> list:
    """
    Convert a raw claim data dict into a feature vector.

    claim_data keys (all optional, with sensible defaults):
        patient_age, gender, hospital_type, disease, bill_amount,
        admission_date, discharge_date, insurance_id (non-empty = valid),
        policy_number (non-empty = valid), verification_score,
        verification_status, previous_claims, fraud_flag

    Returns: list of floats in FEATURE_COLUMNS order.
    """
    age              = parse_age(claim_data.get("patient_age"))
    gender_enc       = encode_gender(claim_data.get("gender", ""))
    hosp_type_enc    = encode_hospital_type(claim_data.get("hospital_type", "private"))
    disease_sev      = encode_disease_severity(claim_data.get("disease", ""))
    bill             = parse_amount(claim_data.get("bill_amount", 0))
    adm_days         = calculate_admission_days(
                           claim_data.get("admission_date"),
                           claim_data.get("discharge_date")
                       )
    ins_valid        = 1 if str(claim_data.get("insurance_id", "")).strip() else 0
    pol_valid        = 1 if str(claim_data.get("policy_number", "")).strip() else 0
    ver_score        = min(100, max(0, int(parse_amount(claim_data.get("verification_score", 50)))))
    ver_status_enc   = encode_verification_status(claim_data.get("verification_status", ""))
    prev_claims      = max(0, int(parse_amount(claim_data.get("previous_claims", 0))))
    fraud_flag       = 1 if str(claim_data.get("fraud_flag", "0")).strip() in ("1", "true", "yes") else 0

    return [
        age, gender_enc, hosp_type_enc, disease_sev,
        bill, adm_days, ins_valid, pol_valid,
        ver_score, ver_status_enc, prev_claims, fraud_flag
    ]