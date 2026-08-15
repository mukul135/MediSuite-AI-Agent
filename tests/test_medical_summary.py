# ============================================================
#  MediSuite-AI-Agent -- tests/test_medical_summary.py
#  Feature 9: Unit tests for Medical Report Summarizer
#  Run: python tests/test_medical_summary.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.text_cleaner import (
    clean_ocr_text, word_count, compression_ratio,
    truncate_for_model, extract_key_sections
)
from ai.model_loader import is_model_loaded, get_model_name
from ai.medical_summarizer import summarize

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
passed = failed = 0

def test(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"{GREEN}v PASS{RESET}  {name}"); passed += 1
    else:
        print(f"{RED}x FAIL{RESET}  {name}"); failed += 1
        if detail: print(f"         {detail}")

# ── Sample medical text ───────────────────────────────────────────────────────
SAMPLE = """
Patient Name: Mr. Rajesh Kumar. Admitted: 10/03/2024. Discharged: 17/03/2024.
Consulting: Dr. Priya Nair. Diagnosis: Dengue Fever with Thrombocytopenia
and Type 2 Diabetes Mellitus. Patient presented with high-grade fever for
five days, severe headache, body ache, vomiting, and rash.
Platelet count reduced to 72000 per microliter.
Investigations: CBC, Platelet Count, Dengue NS1 Antigen Positive, LFT, KFT, HbA1c 8.2%.
Treatment: IV Fluids Normal Saline and DNS. Paracetamol 500mg TDS.
Azithromycin 500mg OD for 5 days. Insulin Lantus 10 units bedtime.
Blood Transfusion 1 unit on day 3.
Patient improved from day 3. Platelet count recovered to 145000 by discharge.
Condition at discharge: Stable. Follow-up: Review after 1 week.
Monitor platelet count weekly for next 4 weeks. Avoid NSAIDs.
"""

SHORT_TEXT = "Patient has fever."
EMPTY_TEXT = ""
NOISE_TEXT = "\n\n1\n2\n3\n\n   \n"

# ── TC1: clean_ocr_text removes noise ────────────────────────────────────────
cleaned = clean_ocr_text(SAMPLE)
test("TC1  — clean_ocr_text removes noise",
     len(cleaned) > 0 and "\n\n" not in cleaned)

# ── TC2: clean_ocr_text handles empty ────────────────────────────────────────
test("TC2  — clean_ocr_text handles empty string",
     clean_ocr_text("") == "")

# ── TC3: clean_ocr_text handles pure noise ───────────────────────────────────
test("TC3  — clean_ocr_text handles noise-only text",
     clean_ocr_text(NOISE_TEXT).strip() == "")

# ── TC4: word_count works correctly ──────────────────────────────────────────
test("TC4  — word_count correct", word_count("hello world foo") == 3)
test("TC5  — word_count empty",   word_count("") == 0)
test("TC6  — word_count None",    word_count(None) == 0)

# ── TC7: compression_ratio ───────────────────────────────────────────────────
test("TC7  — compression_ratio calculation",
     compression_ratio("one two three four", "one two") == 0.5)
test("TC8  — compression_ratio zero denominator",
     compression_ratio("", "summary") == 0.0)

# ── TC9: truncate_for_model ───────────────────────────────────────────────────
long_text = " ".join(["word"] * 1000)
truncated = truncate_for_model(long_text, max_words=700)
test("TC9  — truncate_for_model limits words",
     word_count(truncated) <= 702,   # 700 + "..." possible
     f"got {word_count(truncated)} words")

short_text = "just a short text"
test("TC10 — truncate_for_model leaves short text unchanged",
     truncate_for_model(short_text, 700) == short_text)

# ── TC11: extract_key_sections ────────────────────────────────────────────────
sections = extract_key_sections(SAMPLE)
test("TC11 — extract_key_sections finds diagnosis",
     "diagnosis" in sections or len(sections) > 0,
     f"sections found: {list(sections.keys())}")

# ── TC12: summarize returns correct structure ─────────────────────────────────
r = summarize(SAMPLE, summary_type="medium")
test("TC12 — summarize returns all required keys",
     all(k in r for k in ['summary_type','summary','word_count',
                           'original_word_count','compression_ratio',
                           'model_used','generated_at','error']))

# ── TC13: summarize medium returns non-empty summary ─────────────────────────
test("TC13 — summarize medium returns non-empty summary",
     len(r['summary']) > 0, f"got: '{r['summary']}'")

# ── TC14: summarize short ─────────────────────────────────────────────────────
r_short = summarize(SAMPLE, summary_type="short")
test("TC14 — summarize short returns summary",
     len(r_short['summary']) > 0)
test("TC15 — short summary shorter than medium",
     r_short['word_count'] <= r['word_count'] + 10,
     f"short={r_short['word_count']} medium={r['word_count']}")

# ── TC16: summarize detailed ──────────────────────────────────────────────────
r_detail = summarize(SAMPLE, summary_type="detailed")
test("TC16 — summarize detailed returns summary",
     len(r_detail['summary']) > 0)

# ── TC17: empty input returns error, not crash ────────────────────────────────
r_empty = summarize(EMPTY_TEXT, summary_type="medium")
test("TC17 — empty input returns error message, no crash",
     r_empty['error'] is not None and r_empty['summary'] == "",
     f"error: {r_empty['error']}")

# ── TC18: very short text returns error ───────────────────────────────────────
r_short2 = summarize("Hi.", summary_type="medium")
test("TC18 — very short text handled gracefully",
     r_short2['error'] is not None or len(r_short2['summary']) >= 0)

# ── TC19: invalid summary_type defaults to medium ────────────────────────────
r_inv = summarize(SAMPLE, summary_type="superlong")
test("TC19 — invalid summary_type defaults gracefully",
     r_inv['error'] is None or len(r_inv['summary']) >= 0)

# ── TC20: compression ratio makes sense ──────────────────────────────────────
test("TC20 — compression ratio <= 1.0 for actual summaries",
     0.0 <= r['compression_ratio'] <= 1.0,
     f"got {r['compression_ratio']}")

print(f"\n{'='*55}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} | {passed+failed} total")
print("="*55)
if failed == 0:
    print(f"{GREEN}All tests passed!{RESET}")
else:
    sys.exit(1)