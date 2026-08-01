# ============================================================
#  MediSuite-AI-Agent -- tests/test_fraud_detection.py
#  Feature 8: Unit tests for fraud detection engine
#  Run: python tests/test_fraud_detection.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.fraud_preprocessing import build_fraud_feature_vector, FRAUD_FEATURE_COLUMNS, parse_float
from ml.detect_fraud import detect_fraud, score_to_status, is_fraud_model_ready, get_fraud_model_stats

GREEN="\033[92m"; RED="\033[91m"; RESET="\033[0m"; BOLD="\033[1m"
passed=failed=0

def test(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"{GREEN}v PASS{RESET}  {name}"); passed+=1
    else:
        print(f"{RED}x FAIL{RESET}  {name}"); failed+=1
        if detail: print(f"         {detail}")

# Clean valid claim
VALID = {
    "bill_amount":"48000","insurance_id":"INS/001","policy_number":"POL/001",
    "doctor_name":"Dr. Priya","disease":"Typhoid","verification_score":"92",
    "verification_status":"Eligible","approval_probability":"94",
    "previous_claims":"1","claim_frequency_7d":"0",
    "admission_date":"10/03/2024","discharge_date":"17/03/2024"
}

# ── Preprocessing tests ────────────────────────────────────────────────────────
fv = build_fraud_feature_vector(VALID)
test("TC1 — Feature vector correct length", len(fv)==len(FRAUD_FEATURE_COLUMNS), f"got {len(fv)}")
test("TC2 — parse_float '48,000' = 48000", parse_float("48,000")==48000.0)
test("TC3 — parse_float None = 0.0",       parse_float(None)==0.0)

# ── Score to status ────────────────────────────────────────────────────────────
test("TC4 — score 0  → Low Risk",         score_to_status(0)=="Low Risk")
test("TC5 — score 30 → Medium Risk",      score_to_status(30)=="Medium Risk")
test("TC6 — score 60 → High Risk",        score_to_status(60)=="High Risk")
test("TC7 — score 80 → Fraud Suspected",  score_to_status(80)=="Fraud Suspected")

# ── Rule-based detection ───────────────────────────────────────────────────────
r_clean = detect_fraud(VALID)
test("TC8 — Clean claim → Low Risk",
     r_clean['fraud_status']=='Low Risk', f"got {r_clean['fraud_status']}")
test("TC9 — Clean claim → score < 30",
     r_clean['fraud_score']<30, f"score={r_clean['fraud_score']}")

# Missing insurance ID
r_no_ins = detect_fraud({**VALID,"insurance_id":""})
test("TC10 — No insurance ID → score rises",
     r_no_ins['fraud_score']>0,
     f"score={r_no_ins['fraud_score']}")

# High bill
r_high_bill = detect_fraud({**VALID,"bill_amount":"950000"})
test("TC11 — Very high bill → detected in rules",
     any("Bill" in x or "bill" in x.lower() for x in r_high_bill['detected_rules']),
     f"rules={r_high_bill['detected_rules']}")

# Excessive frequency
r_freq = detect_fraud({**VALID,"claim_frequency_7d":"7"})
test("TC12 — High claim frequency → detected",
     any("7 day" in x or "claims submitted" in x.lower() for x in r_freq['detected_rules']),
     f"rules={r_freq['detected_rules']}")

# Invalid dates
r_dates = detect_fraud({**VALID,"admission_date":"20/03/2024","discharge_date":"10/03/2024"})
test("TC13 — Adm date after dis date → detected",
     any("after" in x.lower() or "Invalid" in x for x in r_dates['detected_rules']),
     f"rules={r_dates['detected_rules']}")

# Verification rejected + AI approved
r_conflict = detect_fraud({**VALID,"verification_status":"Rejected","approval_probability":"88"})
test("TC14 — Verification/AI conflict → detected",
     len(r_conflict['detected_rules'])>0)

# Duplicate detection
r_dup = detect_fraud({**VALID,"past_claims_summary":[{"id":5,"insurance_id":"INS/001","bill_amount":"48000"}]})
test("TC15 — Duplicate claim detected",
     any("Duplicate" in x or "duplicate" in x for x in r_dup['detected_rules']),
     f"rules={r_dup['detected_rules']}")

# Full fraud scenario
FRAUD_DATA = {
    "bill_amount":"950000","insurance_id":"","policy_number":"",
    "doctor_name":"","disease":"","verification_score":"5",
    "verification_status":"Rejected","approval_probability":"82",
    "previous_claims":"9","claim_frequency_7d":"6",
    "admission_date":"20/03/2024","discharge_date":"10/03/2024"
}
r_fraud = detect_fraud(FRAUD_DATA)
test("TC16 — Full fraud scenario → High Risk or Fraud Suspected",
     r_fraud['fraud_status'] in ('High Risk','Fraud Suspected'),
     f"got {r_fraud['fraud_status']} score={r_fraud['fraud_score']}")
test("TC17 — Full fraud → multiple rules triggered",
     len(r_fraud['detected_rules'])>=3,
     f"rules={r_fraud['detected_rules']}")

# Result structure
test("TC18 — Result has all required keys",
     all(k in r_clean for k in ['fraud_status','fraud_score','fraud_probability',
                                  'detected_rules','anomaly_detected','recommendation',
                                  'model_name','timestamp']))

# Model file
test("TC19 — Fraud model file exists", is_fraud_model_ready(), "Run: python ml/train_fraud_model.py")

# Stats
if is_fraud_model_ready():
    stats = get_fraud_model_stats()
    test("TC20 — Model stats loadable", 'mean_bill' in stats)
else:
    print("  TC20 — SKIPPED")

print(f"\n{'='*55}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} | {passed+failed} total")
print("="*55)
if failed==0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    sys.exit(1)