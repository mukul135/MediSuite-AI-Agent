# ============================================================
#  MediSuite-AI-Agent -- test_extractor.py
#  Phase 4: Testing strategy for the claim extraction engine
#
#  Run with:  python test_extractor.py
#  All tests print PASS / FAIL clearly.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor.claim_extractor import extract_claim_fields

# ── Colour output for terminal ──────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = 0
failed = 0

def test(name, ocr_text, expected):
    """
    Run one test case.
    expected = dict with keys: patient_name, hospital_name, disease, bill_amount
    Each value is either an exact string or a substring that must appear in result.
    """
    global passed, failed

    result = extract_claim_fields(ocr_text)
    all_pass = True
    failures = []

    for field, expected_val in expected.items():
        actual = result.get(field, "").strip().lower()
        if expected_val.lower() not in actual and actual not in expected_val.lower():
            all_pass = False
            failures.append(f"  {field}: expected '{expected_val}', got '{result.get(field, '')}'")

    if all_pass:
        print(f"{GREEN}✓ PASS{RESET}  {name}")
        passed += 1
    else:
        print(f"{RED}✗ FAIL{RESET}  {name}")
        for f in failures:
            print(f"         {f}")
        failed += 1


# ============================================================
#  TEST CASES
# ============================================================

# ── TC1: Clean, well-formatted discharge summary ─────────────────────────────
test("TC1 — Clean discharge summary",
    """
    APOLLO HOSPITALS LIMITED
    Hospital Name: Apollo Hospital, Chennai

    Patient Name: Mr. Rajesh Kumar
    Diagnosis: Typhoid Fever  ICD-10: A01.0

    Grand Total: Rs. 48,150
    """,
    {
        "patient_name":  "Rajesh Kumar",
        "hospital_name": "Apollo",
        "disease":       "Typhoid",
        "bill_amount":   "48,150"
    }
)

# ── TC2: Noisy OCR output with typos ─────────────────────────────────────────
test("TC2 — Noisy OCR with misspellings",
    """
    FORTIS HOSPIT AL LTD

    Patlent Narne : Mrs. Priya Sharma
    Diagnoiss : Type 2 Diabetes Mellitus
    Totall Amount : Rs 35000
    """,
    {
        "patient_name":  "Priya Sharma",
        "hospital_name": "Fortis",
        "disease":       "Diabetes",
        "bill_amount":   "35000"
    }
)

# ── TC3: ICD-10 code present ──────────────────────────────────────────────────
test("TC3 — ICD-10 code detection",
    """
    MANIPAL HOSPITAL
    Patient: Anil Verma  Age: 60
    Final Diagnosis: Pneumonia  J18.9
    Total Bill: INR 72,000
    """,
    {
        "patient_name":  "Anil Verma",
        "hospital_name": "Manipal",
        "disease":       "Pneumonia",
        "bill_amount":   "72,000"
    }
)

# ── TC4: Rupee symbol ₹ ───────────────────────────────────────────────────────
test("TC4 — Rupee symbol amount",
    """
    AIIMS New Delhi

    Name of Patient: Dr. Sunita Patel
    Chief Complaint: Hypertension with Cardiac complications

    Grand Total  ₹ 1,25,000
    """,
    {
        "patient_name":  "Sunita Patel",
        "hospital_name": "AIIMS",
        "disease":       "Hypertension",
        "bill_amount":   "1,25,000"
    }
)

# ── TC5: All caps patient name ────────────────────────────────────────────────
test("TC5 — ALL CAPS patient name",
    """
    MAX SUPER SPECIALTY HOSPITAL

    PATIENT NAME: KUMAR RAVI SHANKAR
    DIAGNOSIS: DENGUE FEVER
    TOTAL AMOUNT: RS. 28500
    """,
    {
        "patient_name":  "Kumar Ravi",
        "hospital_name": "Max",
        "disease":       "Dengue",
        "bill_amount":   "28500"
    }
)

# ── TC6: No patient name — should return empty ────────────────────────────────
test("TC6 — Missing patient name field",
    """
    Some Generic Clinic

    Invoice #: 4521
    Disease: Malaria
    Bill: Rs. 5000
    """,
    {
        "hospital_name": "Clinic",
        "disease":       "Malaria",
        "bill_amount":   "5000"
    }
)

# ── TC7: Multiple amounts — should pick largest (grand total) ─────────────────
test("TC7 — Multiple amounts, pick grand total",
    """
    Narayana Hospital Bangalore

    Patient: Mr. Deepak Nair
    Condition: Kidney Failure

    Room Charges        Rs. 12,000
    Doctor Fees         Rs.  8,000
    Medicines           Rs.  9,500
    Lab Tests           Rs.  6,200
    GRAND TOTAL         Rs. 35,700
    """,
    {
        "patient_name":  "Deepak Nair",
        "hospital_name": "Narayana",
        "disease":       "Kidney",
        "bill_amount":   "35,700"
    }
)

# ── TC8: Very minimal document ────────────────────────────────────────────────
test("TC8 — Minimal document",
    """
    Patient: Meera Iyer
    Diagnosis: Fever
    Amount: 2000
    """,
    {
        "patient_name": "Meera Iyer",
        "disease":      "Fever",
        "bill_amount":  "2000"
    }
)


# ============================================================
#  SUMMARY
# ============================================================

print(f"\n{'='*50}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}  |  {passed+failed} total")
print(f"{'='*50}")

if failed == 0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    print(f"{RED}Some tests failed. Review extraction logic.{RESET}")
    sys.exit(1)