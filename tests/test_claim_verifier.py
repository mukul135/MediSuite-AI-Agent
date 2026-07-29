# ============================================================
#  MediSuite-AI-Agent -- test_claim_verifier.py
#  Feature 6: Unit tests for claim_verifier.py
#  Run: python test_claim_verifier.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor.claim_verifier import verify_claim

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
passed = failed = 0

# Full valid fields — base for most tests
FULL_VALID = {
    "patient_name":   "Mr. Rajesh Kumar",
    "hospital_name":  "Apollo Hospital Chennai",
    "disease":        "Typhoid Fever",
    "bill_amount":    "48150",
    "insurance_id":   "INS/2024/00123",
    "policy_number":  "POL/2024/00456",
    "admission_date": "10/03/2024",
    "discharge_date": "17/03/2024",
    "doctor_name":    "Dr. Priya Nair"
}


def test(name, fields, expected_status, extra_checks=None):
    global passed, failed
    result = verify_claim(fields)
    errors = []
    if result['status'] != expected_status:
        errors.append(f"  Status: expected '{expected_status}', got '{result['status']}'")
    if extra_checks:
        for check_fn, check_desc in extra_checks:
            if not check_fn(result):
                errors.append(f"  Check failed: {check_desc}")
    if not errors:
        print(f"{GREEN}v PASS{RESET}  {name}  [Status: {result['status']}  Score: {result['score']}%]")
        passed += 1
    else:
        print(f"{RED}x FAIL{RESET}  {name}")
        for e in errors: print(e)
        failed += 1


# ── TC1: All fields valid → Eligible ─────────────────────────────────────────
test("TC1 — All fields valid → Eligible",
     FULL_VALID,
     "Eligible",
     [(lambda r: r['score'] >= 80, "Score should be >= 80%"),
      (lambda r: len(r['failed_rules']) == 0, "No failed rules"),
      (lambda r: len(r['missing_fields']) == 0, "No missing fields")])

# ── TC2: Insurance ID missing → Rejected ─────────────────────────────────────
test("TC2 — Insurance ID Missing → Rejected",
     {**FULL_VALID, "insurance_id": ""},
     "Rejected",
     [(lambda r: any("Insurance ID" in f for f in r['failed_rules']), "Insurance ID in failed rules")])

# ── TC3: Policy number missing → Incomplete ───────────────────────────────────
test("TC3 — Policy Number Missing → Incomplete",
     {**FULL_VALID, "insurance_id": "INS123", "policy_number": ""},
     "Incomplete",
     [(lambda r: any("Policy" in f for f in r['failed_rules']), "Policy in failed rules")])

# ── TC4: Hospital name missing → Incomplete ───────────────────────────────────
test("TC4 — Hospital Name Missing → Incomplete",
     {**FULL_VALID, "hospital_name": ""},
     "Incomplete",
     [(lambda r: "Hospital Name" in r['missing_fields'], "Hospital Name in missing fields")])

# ── TC5: Disease missing → Incomplete ────────────────────────────────────────
test("TC5 — Disease Missing → Incomplete",
     {**FULL_VALID, "disease": ""},
     "Incomplete",
     [(lambda r: "Disease" in r['missing_fields'], "Disease in missing fields")])

# ── TC6: Bill amount zero → Rejected ─────────────────────────────────────────
test("TC6 — Bill Amount Zero → Rejected",
     {**FULL_VALID, "bill_amount": "0"},
     "Rejected",
     [(lambda r: any("Zero" in f or "Greater" in f for f in r['failed_rules']),
       "Bill amount zero in failed rules")])

# ── TC7: Admission date AFTER discharge → Rejected ───────────────────────────
test("TC7 — Invalid Dates (Adm > Dis) → Rejected",
     {**FULL_VALID, "admission_date": "20/03/2024", "discharge_date": "10/03/2024"},
     "Rejected",
     [(lambda r: any("Invalid Dates" in f or "AFTER" in f for f in r['failed_rules']),
       "Invalid date in failed rules")])

# ── TC8: Multiple missing fields → Incomplete ─────────────────────────────────
test("TC8 — Multiple Missing Fields → Rejected (no Insurance ID)",
     {"patient_name": "John Doe", "bill_amount": "5000"},
     "Rejected",
     [(lambda r: len(r['missing_fields']) >= 3, "At least 3 fields missing")])

# ── TC9: Empty dict → Rejected (insurance ID missing) ────────────────────────
test("TC9 — Empty Fields → Rejected",
     {},
     "Rejected",
     [(lambda r: r['score'] < 60, "Low score for empty fields")])

# ── TC10: Score present and in range 0–100 ────────────────────────────────────
test("TC10 — Score Always 0–100",
     FULL_VALID,
     "Eligible",
     [(lambda r: 0 <= r['score'] <= 100, "Score in range 0–100"),
      (lambda r: r['verified_at'] is not None, "verified_at timestamp present"),
      (lambda r: r['remarks'] != "", "Remarks not empty")])


print(f"\n{'='*55}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} | {passed+failed} total")
print("="*55)
if failed == 0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    sys.exit(1)