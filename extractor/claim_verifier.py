# ============================================================
#  MediSuite-AI-Agent -- extractor/claim_verifier.py
#  Feature 6: Insurance Claim Verification Engine
#
#  Pure rule-based verification. No ML, no external APIs.
#  Completely independent of claim_extractor.py and medical_nlp.py
#
#  Input  : dict of claim fields
#  Output : structured verification result dict
# ============================================================

from datetime import datetime
import re


# ============================================================
#  SECTION 1: RULE DEFINITIONS
#  severity: "reject" = hard fail, "incomplete" = soft fail
# ============================================================

RULES = [
    {"id": "R01", "desc": "Insurance ID Present",          "severity": "reject"},
    {"id": "R02", "desc": "Policy Number Present",         "severity": "incomplete"},
    {"id": "R03", "desc": "Patient Name Present",          "severity": "incomplete"},
    {"id": "R04", "desc": "Hospital Name Present",         "severity": "incomplete"},
    {"id": "R05", "desc": "Disease / Diagnosis Present",   "severity": "incomplete"},
    {"id": "R06", "desc": "Bill Amount Present",           "severity": "incomplete"},
    {"id": "R07", "desc": "Bill Amount Greater Than Zero", "severity": "reject"},
    {"id": "R08", "desc": "Valid Admission/Discharge Dates","severity": "reject"},
    {"id": "R09", "desc": "Doctor Name Present",           "severity": "incomplete"},
    {"id": "R10", "desc": "Bill Amount Within Limit",      "severity": "reject"},
]

MAX_BILL_AMOUNT = 10_000_000  # 1 crore INR
MIN_BILL_AMOUNT = 1


# ============================================================
#  SECTION 2: FIELD VALIDATORS
# ============================================================

def _has_value(val) -> bool:
    if val is None:
        return False
    return str(val).strip() != ""


def _parse_amount(amount_str) -> float:
    if amount_str is None:
        return -1
    try:
        cleaned = str(amount_str).replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return -1


def _parse_date(date_str):
    if not date_str or not str(date_str).strip():
        return None
    date_str = str(date_str).strip()
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d %b %Y", "%d %B %Y", "%Y/%m/%d",
        "%d/%m/%y", "%d-%m-%y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# ============================================================
#  SECTION 3: INDIVIDUAL RULE CHECKS
# ============================================================

def check_insurance_id(fields):
    if _has_value(fields.get("insurance_id")):
        return True, "Insurance ID Present"
    return False, "Insurance ID Missing — claim cannot proceed without valid Insurance ID"


def check_policy_number(fields):
    if _has_value(fields.get("policy_number")):
        return True, "Policy Number Present"
    return False, "Policy Number Missing — required for claim processing"


def check_patient_name(fields):
    if _has_value(fields.get("patient_name")):
        return True, "Patient Name Present"
    return False, "Patient Name Missing"


def check_hospital_name(fields):
    if _has_value(fields.get("hospital_name")):
        return True, "Hospital Name Present"
    return False, "Hospital Name Missing"


def check_disease(fields):
    if _has_value(fields.get("disease")):
        return True, "Disease / Diagnosis Present"
    return False, "Disease / Diagnosis Missing"


def check_bill_amount_present(fields):
    if _has_value(fields.get("bill_amount")):
        return True, "Bill Amount Present"
    return False, "Bill Amount Missing"


def check_bill_amount_positive(fields):
    amount = _parse_amount(fields.get("bill_amount"))
    if amount < 0:
        return True, "Bill Amount Not Provided (skipping positivity check)"
    if amount > MIN_BILL_AMOUNT:
        return True, f"Bill Amount Valid (Rs. {amount:,.2f})"
    return False, f"Bill Amount Must Be Greater Than Zero (got: {amount})"


def check_dates(fields):
    adm_str = fields.get("admission_date")
    dis_str = fields.get("discharge_date")
    if not _has_value(adm_str) or not _has_value(dis_str):
        return True, "Date Validation Skipped (dates not provided)"
    adm = _parse_date(adm_str)
    dis = _parse_date(dis_str)
    if adm is None or dis is None:
        return False, "Invalid Date Format — use DD/MM/YYYY"
    if adm <= dis:
        return True, f"Valid Dates: Admission {adm.strftime('%d %b %Y')} to Discharge {dis.strftime('%d %b %Y')}"
    return False, f"Invalid Dates: Admission ({adm.strftime('%d %b %Y')}) is AFTER Discharge ({dis.strftime('%d %b %Y')})"


def check_doctor_name(fields):
    if _has_value(fields.get("doctor_name")):
        return True, "Doctor Name Present"
    return False, "Doctor Name Not Provided (recommended)"


def check_bill_limit(fields):
    amount = _parse_amount(fields.get("bill_amount"))
    if amount < 0:
        return True, "Bill Limit Check Skipped"
    if amount <= MAX_BILL_AMOUNT:
        return True, f"Bill Amount Within Limit (Rs. {amount:,.2f})"
    return False, f"Bill Amount Exceeds Maximum Claimable Limit (Rs. {amount:,.2f} > Rs. {MAX_BILL_AMOUNT:,})"


# ============================================================
#  SECTION 4: RULE RUNNER MAP
# ============================================================

RULE_FUNCTIONS = {
    "R01": check_insurance_id,
    "R02": check_policy_number,
    "R03": check_patient_name,
    "R04": check_hospital_name,
    "R05": check_disease,
    "R06": check_bill_amount_present,
    "R07": check_bill_amount_positive,
    "R08": check_dates,
    "R09": check_doctor_name,
    "R10": check_bill_limit,
}


# ============================================================
#  SECTION 5: SCORE CALCULATOR
# ============================================================

def calculate_score(rule_results):
    """
    Weighted score: reject rules worth 15pts, incomplete = 8pts.
    Score = (earned / max) * 100
    """
    WEIGHTS = {"reject": 15, "incomplete": 8}
    max_pts = sum(WEIGHTS.get(r["severity"], 5) for r in RULES)
    earned  = sum(
        WEIGHTS.get(rule["severity"], 5)
        for rule, passed, _ in rule_results if passed
    )
    return min(100, int((earned / max_pts) * 100)) if max_pts > 0 else 0


# ============================================================
#  SECTION 6: STATUS DETERMINER
#  ANY reject failure → Rejected
#  ANY incomplete failure → Incomplete
#  All pass → Eligible
# ============================================================

def determine_status(rule_results):
    for rule, passed, _ in rule_results:
        if not passed and rule["severity"] == "reject":
            return "Rejected"
    for rule, passed, _ in rule_results:
        if not passed and rule["severity"] == "incomplete":
            return "Incomplete"
    return "Eligible"


# ============================================================
#  SECTION 7: REMARKS GENERATOR
# ============================================================

def generate_remarks(status, missing_fields, failed_rules):
    if status == "Eligible":
        return "All verification checks passed. Claim is eligible for insurance processing."
    if status == "Rejected":
        if failed_rules:
            return f"Claim REJECTED. Critical issue: {failed_rules[0]}."
        return "Claim REJECTED due to critical validation failure."
    if missing_fields:
        return f"Claim INCOMPLETE. Missing: {', '.join(missing_fields[:4])}. Please complete all required fields."
    return "Claim INCOMPLETE. Some required information is missing or invalid."


# ============================================================
#  SECTION 8: MISSING FIELDS DETECTOR
# ============================================================

def _detect_missing_fields(fields):
    required = [
        "patient_name", "hospital_name", "disease",
        "bill_amount", "insurance_id", "policy_number"
    ]
    return [f.replace("_", " ").title() for f in required if not _has_value(fields.get(f))]


# ============================================================
#  SECTION 9: MAIN VERIFICATION PIPELINE
# ============================================================

def verify_claim(fields: dict) -> dict:
    """
    Main entry point for claim verification.

    Input fields dict (all keys optional but recommended):
      patient_name, hospital_name, disease, bill_amount,
      insurance_id, policy_number, admission_date,
      discharge_date, doctor_name

    Returns structured result dict.
    """
    rule_results = []
    for rule in RULES:
        fn = RULE_FUNCTIONS.get(rule["id"])
        if fn:
            try:
                passed, reason = fn(fields)
            except Exception as e:
                passed, reason = False, f"Rule check error: {str(e)}"
            rule_results.append((rule, passed, reason))

    passed_rules   = [reason for _, passed, reason in rule_results if passed]
    failed_rules   = [reason for _, passed, reason in rule_results if not passed]
    missing_fields = _detect_missing_fields(fields)
    status         = determine_status(rule_results)
    score          = calculate_score(rule_results)
    remarks        = generate_remarks(status, missing_fields, failed_rules)

    return {
        "status":         status,
        "score":          score,
        "missing_fields": missing_fields,
        "failed_rules":   failed_rules,
        "passed_rules":   passed_rules,
        "remarks":        remarks,
        "verified_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ============================================================
#  SECTION 10: FIELDS BUILDER FROM DB RECORDS
# ============================================================

def build_fields_from_records(claim_record=None, form_data=None):
    """
    Build fields dict from claim DB record and/or form submission.
    form_data values override claim_record values.
    """
    fields = {}
    if claim_record:
        fields.update({
            "patient_name":  claim_record.get("patient_name", ""),
            "hospital_name": claim_record.get("hospital_name", ""),
            "disease":       claim_record.get("disease", ""),
            "bill_amount":   claim_record.get("bill_amount", ""),
            "policy_number": claim_record.get("policy_number", ""),
        })
    if form_data:
        for key in ["patient_name", "hospital_name", "disease", "bill_amount",
                    "insurance_id", "policy_number", "admission_date",
                    "discharge_date", "doctor_name"]:
            val = form_data.get(key, "")
            if val and str(val).strip():
                fields[key] = val
    return fields


# ============================================================
#  STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("TEST 1: ELIGIBLE CLAIM")
    r = verify_claim({
        "patient_name": "Mr. Rajesh Kumar",
        "hospital_name": "Apollo Hospital",
        "disease": "Typhoid Fever",
        "bill_amount": "48150",
        "insurance_id": "INS/2024/00123",
        "policy_number": "POL/2024/00456",
        "admission_date": "10/03/2024",
        "discharge_date": "17/03/2024",
        "doctor_name": "Dr. Priya Nair"
    })
    print(f"Status: {r['status']}  Score: {r['score']}%")
    print(f"Remarks: {r['remarks']}")

    print("\n" + "=" * 55)
    print("TEST 2: MISSING INSURANCE ID → REJECTED")
    r2 = verify_claim({"patient_name": "John", "hospital_name": "City Hospital",
                        "disease": "Fever", "bill_amount": "5000"})
    print(f"Status: {r2['status']}  Score: {r2['score']}%")
    print(f"Failed: {r2['failed_rules']}")

    print("\n" + "=" * 55)
    print("TEST 3: INCOMPLETE CLAIM")
    r3 = verify_claim({"patient_name": "Jane"})
    print(f"Status: {r3['status']}  Missing: {r3['missing_fields']}")