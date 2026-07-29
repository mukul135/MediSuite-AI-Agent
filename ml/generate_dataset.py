# ============================================================
#  MediSuite-AI-Agent -- ml/generate_dataset.py
#  Feature 7: Synthetic Insurance Claim Dataset Generator
#
#  Generates realistic synthetic insurance claim data aligned
#  with MediSuite's fields.
#
#  Run: python ml/generate_dataset.py
#  Output: dataset/insurance_claim_dataset.csv
# ============================================================

import os
import random
import csv
from datetime import datetime, timedelta

# ── Reproducible randomness ───────────────────────────────────────────────────
random.seed(42)

NUM_RECORDS = 2000

# ── Reference data ───────────────────────────────────────────────────────────
DISEASES = [
    ("Typhoid Fever", "medium"), ("Dengue Fever", "medium"),
    ("Diabetes Mellitus", "medium"), ("Hypertension", "medium"),
    ("Pneumonia", "medium"), ("Tuberculosis", "medium"),
    ("Appendicitis", "medium"), ("Fracture", "low"),
    ("Viral Fever", "low"), ("Gastritis", "low"),
    ("Cardiac Arrest", "high"), ("Heart Failure", "high"),
    ("Kidney Failure", "high"), ("Cancer", "high"),
    ("Stroke", "high"), ("Sepsis", "high"),
    ("Malaria", "low"), ("Asthma", "medium"),
    ("Hepatitis B", "medium"), ("Dengue with Thrombocytopenia", "high"),
]

HOSPITALS = [
    ("Apollo Hospital", "private"),
    ("AIIMS Delhi", "government"),
    ("Fortis Healthcare", "private"),
    ("Narayana Hospital", "private"),
    ("Max Healthcare", "private"),
    ("Government District Hospital", "government"),
    ("City Trust Hospital", "trust"),
    ("Manipal Hospital", "private"),
    ("Ruby Hall Clinic", "private"),
    ("ESI Hospital", "government"),
]

GENDERS = ["male", "female"]

SEVERITY_BILL = {
    "low":    (3000,  25000),
    "medium": (15000, 80000),
    "high":   (50000, 500000),
}

SEVERITY_DAYS = {
    "low":    (1, 4),
    "medium": (3, 10),
    "high":   (7, 30),
}


def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_record(i: int) -> dict:
    """Generate one synthetic insurance claim record."""

    # Patient demographics
    age    = random.randint(18, 80)
    gender = random.choice(GENDERS)

    # Disease and hospital
    disease_name, severity = random.choice(DISEASES)
    hospital_name, hosp_type = random.choice(HOSPITALS)

    # Financials
    bill_min, bill_max = SEVERITY_BILL[severity]
    bill_amount = round(random.uniform(bill_min, bill_max), 2)

    # Dates
    adm_date = random_date()
    day_min, day_max = SEVERITY_DAYS[severity]
    adm_days = random.randint(day_min, day_max)
    dis_date = adm_date + timedelta(days=adm_days)

    # Insurance completeness (realistic distribution)
    insurance_id_valid = 1 if random.random() > 0.15 else 0
    policy_valid       = 1 if random.random() > 0.20 else 0

    # Verification (correlated with completeness)
    if insurance_id_valid and policy_valid:
        ver_score  = random.randint(60, 100)
        ver_status = "Eligible" if ver_score >= 80 else "Incomplete"
    elif insurance_id_valid or policy_valid:
        ver_score  = random.randint(30, 75)
        ver_status = "Incomplete"
    else:
        ver_score  = random.randint(0, 40)
        ver_status = "Rejected"

    previous_claims = random.randint(0, 5)
    fraud_flag      = 1 if random.random() < 0.05 else 0  # 5% fraud rate

    # ── LABEL GENERATION ─────────────────────────────────────────────────────
    # Approval logic (rule-based generation, realistic but not perfect):
    # - Must have insurance + policy
    # - Verification must be Eligible or high-score Incomplete
    # - No fraud flag
    # - High bill with low verification → likely rejected
    # - Some randomness to simulate real-world uncertainty

    score = 0
    if insurance_id_valid:     score += 30
    if policy_valid:           score += 20
    if ver_status == "Eligible": score += 25
    elif ver_status == "Incomplete": score += 10
    score += min(ver_score // 5, 20)
    if not fraud_flag:         score += 10
    if previous_claims <= 2:   score += 5

    # Add small noise
    score += random.randint(-10, 10)
    score = max(0, min(100, score))

    approved = 1 if score >= 55 else 0

    return {
        "patient_age":        age,
        "gender":             gender,
        "hospital_type":      hosp_type,
        "disease":            disease_name,
        "disease_severity":   severity,
        "bill_amount":        bill_amount,
        "admission_days":     adm_days,
        "admission_date":     adm_date.strftime("%d/%m/%Y"),
        "discharge_date":     dis_date.strftime("%d/%m/%Y"),
        "insurance_id_valid": insurance_id_valid,
        "policy_valid":       policy_valid,
        "verification_score": ver_score,
        "verification_status": ver_status,
        "previous_claims":    previous_claims,
        "fraud_flag":         fraud_flag,
        "approved":           approved,  # TARGET LABEL: 1=Approved, 0=Rejected
    }


def generate_dataset(output_path: str, n: int = NUM_RECORDS):
    """Generate and save the dataset to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records  = [generate_record(i) for i in range(n)]
    fieldnames = list(records[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    approved_count = sum(1 for r in records if r["approved"] == 1)
    print(f"Dataset generated: {output_path}")
    print(f"  Total records : {n}")
    print(f"  Approved      : {approved_count} ({approved_count/n*100:.1f}%)")
    print(f"  Rejected      : {n - approved_count} ({(n-approved_count)/n*100:.1f}%)")


if __name__ == "__main__":
    output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "insurance_claim_dataset.csv"
    )
    generate_dataset(output)