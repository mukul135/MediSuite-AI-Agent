# ============================================================
#  MediSuite-AI-Agent -- tests/test_claim_prediction.py
#  Feature 7: Unit tests for ML prediction pipeline
#  Run: python tests/test_claim_prediction.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import build_feature_vector, FEATURE_COLUMNS, parse_amount, parse_age
from ml.predict_claim  import is_model_ready, classify_confidence, predict, get_model_metrics

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
passed = failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"{GREEN}v PASS{RESET}  {name}")
        passed += 1
    else:
        print(f"{RED}x FAIL{RESET}  {name}")
        if detail: print(f"         {detail}")
        failed += 1


# ── TC1: Feature vector length matches FEATURE_COLUMNS ───────────────────────
fv = build_feature_vector({"patient_age": 45, "gender": "male"})
test("TC1 — Feature vector correct length",
     len(fv) == len(FEATURE_COLUMNS),
     f"expected {len(FEATURE_COLUMNS)}, got {len(fv)}")

# ── TC2: parse_amount handles strings correctly ───────────────────────────────
test("TC2 — parse_amount: '48,150' → 48150.0", parse_amount("48,150") == 48150.0)
test("TC3 — parse_amount: '' → 0.0",           parse_amount("") == 0.0)
test("TC4 — parse_amount: None → 0.0",         parse_amount(None) == 0.0)

# ── TC5: parse_age ────────────────────────────────────────────────────────────
test("TC5 — parse_age: '45' → 45",  parse_age("45") == 45)
test("TC6 — parse_age: None → 35",  parse_age(None) == 35)
test("TC7 — parse_age: '999' → 120", parse_age("999") == 120)

# ── TC8: Confidence classifier ────────────────────────────────────────────────
test("TC8 — classify_confidence 95 → High",   classify_confidence(95) == "High")
test("TC9 — classify_confidence 75 → Medium", classify_confidence(75) == "Medium")
test("TC10 — classify_confidence 50 → Low",   classify_confidence(50) == "Low")

# ── TC11: Model file exists ───────────────────────────────────────────────────
model_ready = is_model_ready()
test("TC11 — Model file exists",
     model_ready,
     "Run: python ml/train_model.py")

# ── TC12–TC16: Full prediction tests (only if model trained) ─────────────────
if model_ready:
    # Eligible claim → should be Approved
    r = predict({
        "patient_age": 45, "gender": "male", "hospital_type": "private",
        "disease": "Typhoid Fever", "bill_amount": "48150",
        "admission_date": "10/03/2024", "discharge_date": "17/03/2024",
        "insurance_id": "INS/001", "policy_number": "POL/001",
        "verification_score": 95, "verification_status": "Eligible",
        "previous_claims": 0, "fraud_flag": 0
    })
    test("TC12 — Eligible claim → Approved",
         r['prediction'] == 'Approved',
         f"Got: {r['prediction']} ({r['approval_probability']}%)")

    test("TC13 — Approval probability 0–100",
         0 <= r['approval_probability'] <= 100,
         f"Got: {r['approval_probability']}")

    test("TC14 — Probabilities sum to 100",
         abs(r['approval_probability'] + r['rejection_probability'] - 100) < 0.5,
         f"Sum: {r['approval_probability'] + r['rejection_probability']}")

    test("TC15 — Result has all required keys",
         all(k in r for k in ['prediction','approval_probability','rejection_probability','confidence','model_name','prediction_time']))

    # Rejected claim (no insurance, fraud flag)
    r2 = predict({
        "patient_age": 35, "disease": "Cancer", "bill_amount": "450000",
        "insurance_id": "", "policy_number": "",
        "verification_score": 10, "verification_status": "Rejected",
        "previous_claims": 5, "fraud_flag": 1
    })
    test("TC16 — Rejected claim → Rejected",
         r2['prediction'] == 'Rejected',
         f"Got: {r2['prediction']} ({r2['rejection_probability']}% rejection)")

else:
    print(f"  (Skipping TC12–TC16: model not trained)")
    for i in range(12, 17):
        print(f"  TC{i} — SKIPPED (model not ready)")

# ── TC17: get_model_metrics ───────────────────────────────────────────────────
if model_ready:
    metrics = get_model_metrics()
    test("TC17 — Model metrics loadable", isinstance(metrics, dict) and 'accuracy' in metrics)
else:
    print(f"  TC17 — SKIPPED")

print(f"\n{'='*55}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} | {passed+failed} total")
print("="*55)
if failed == 0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    sys.exit(1)