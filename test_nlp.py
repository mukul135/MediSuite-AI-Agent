# ============================================================
#  MediSuite-AI-Agent -- test_nlp.py
#  Feature 5: Unit tests for medical_nlp.py
#  Run: python test_nlp.py
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor.medical_nlp import analyze_medical_text

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
passed = failed = 0

def test(name, ocr_text, must_contain):
    global passed, failed
    result = analyze_medical_text(ocr_text)
    errors = []
    for field, keywords in must_contain.items():
        actual = [v.lower() for v in result.get(field, [])]
        for kw in keywords:
            if not any(kw.lower() in a for a in actual):
                errors.append(f"  {field}: expected '{kw}' not found (got: {result.get(field,[])})")
    if not errors:
        print(f"{GREEN}v PASS{RESET}  {name}"); passed += 1
    else:
        print(f"{RED}x FAIL{RESET}  {name}")
        for e in errors: print(e)
        failed += 1

test("TC1 - Disease detection",
    "Patient diagnosed with Dengue Fever and Type 2 Diabetes Mellitus.",
    {"diseases": ["Dengue", "Diabetes"]})

test("TC2 - Medicine detection",
    "Prescribed: Paracetamol 500mg TDS, Azithromycin 500mg OD, Pantoprazole 40mg BD",
    {"medicines": ["Paracetamol", "Azithromycin", "Pantoprazole"]})

test("TC3 - Treatment detection",
    "Patient underwent Blood Transfusion and received IV Fluids. ECG done.",
    {"treatments": ["Blood Transfusion", "IV Fluids"]})

test("TC4 - Symptom detection",
    "Complaints: High Fever, Vomiting, Body ache, Headache, Rash",
    {"symptoms": ["Fever", "Vomiting", "Headache"]})

test("TC5 - Test detection",
    "Investigations: CBC, LFT, KFT, HbA1c, Dengue NS1 Antigen, Platelet Count",
    {"tests": ["CBC", "LFT"]})

test("TC6 - Doctor detection",
    "Consulting Physician: Dr. Priya Nair\nSurgeon: Dr. Anil Mehta",
    {"doctors": ["Dr. Priya Nair", "Dr. Anil Mehta"]})

test("TC7 - Noisy OCR text",
    "Diagnoiss: Typhoyd Fevr\nMedicne: Paracetamol 500mg\nTreatmnt: Blood Test",
    {"medicines": ["Paracetamol"]})

test("TC8 - Mixed complex report",
    """DISCHARGE SUMMARY
    Patient: Mr. Rajesh Kumar  Consulting: Dr. Suresh Iyer
    Diagnosis: Pneumonia with Sepsis
    Complaints: Fever, Breathlessness, Cough
    Treatment: IV Fluids, Oxygen Therapy, Blood Culture
    Medicines: Ceftriaxone 1g BD, Metronidazole 500mg TDS, Paracetamol 650mg SOS
    Tests: CBC, CRP, Blood Culture, X-Ray Chest, CT Chest""",
    {"diseases": ["Pneumonia","Sepsis"], "symptoms": ["Fever","Cough"],
     "medicines": ["Ceftriaxone","Metronidazole","Paracetamol"], "doctors": ["Dr. Suresh"]})

test("TC9 - Empty text returns no error crash",
    "",
    {})

test("TC10 - Feature 4 unaffected (extractor still importable)",
    "dummy", {})

print(f"\n{'='*50}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} | {passed+failed} total")
print("="*50)
if failed == 0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    sys.exit(1)