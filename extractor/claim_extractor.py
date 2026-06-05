# ============================================================
#  MediSuite-AI-Agent -- extractor/claim_extractor.py
#  Phase 4: Smart Insurance Claim Form Autofill
#
#  This module takes raw OCR text and extracts:
#    - Patient Name
#    - Hospital Name
#    - Disease / Diagnosis
#    - Bill Amount
#
#  Approach:
#    1. Regex patterns      -- fast, rule-based extraction
#    2. Keyword context     -- look for labels then grab nearby values
#    3. Fuzzy matching      -- handle OCR misspellings
#    4. Confidence scoring  -- rank multiple candidates, pick best
# ============================================================

import re
import json
from difflib import SequenceMatcher


# ============================================================
#  SECTION 1: CONSTANTS & KEYWORD DICTIONARIES
#  These are the "hints" we look for in OCR text to find fields.
#  Keeping them here (not scattered through code) = easy to update.
# ============================================================

# Labels that typically appear before a patient's name in medical docs
PATIENT_LABELS = [
    "patient name", "patient's name", "name of patient", "patient",
    "pt. name", "pt name", "patient details",
    "admitted patient", "name of insured", "beneficiary name",
    "member name", "claimant name"
    # NOTE: bare "name" removed — too generic, matches "Hospital Name:" lines
]

# Labels that appear before a hospital/clinic name
HOSPITAL_LABELS = [
    "hospital name", "hospital", "clinic", "medical centre", "medical center",
    "health centre", "health center", "name of hospital", "facility name",
    "treatment facility", "treating hospital", "institution", "nursing home",
    "healthcare facility", "diagnostic centre", "diagnostic center"
]

# Labels that appear before a disease or diagnosis
DISEASE_LABELS = [
    "diagnosis", "disease", "condition", "ailment", "illness", "disorder",
    "chief complaint", "presenting complaint", "primary diagnosis",
    "secondary diagnosis", "final diagnosis", "provisional diagnosis",
    "icd code", "icd-10", "clinical diagnosis", "medical condition",
    "reason for admission", "reason for hospitalization", "discharge diagnosis"
]

# Labels that appear before an amount in bills
AMOUNT_LABELS = [
    "total amount", "total bill", "grand total", "net amount", "amount payable",
    "total payable", "bill amount", "total charges", "total cost",
    "amount due", "net payable", "invoice total", "final amount",
    "total invoice", "amount", "rs.", "inr", "₹", "usd", "$"
]

# Common Indian hospital name suffixes — used during entity detection
HOSPITAL_SUFFIXES = [
    "hospital", "hospitals", "clinic", "clinics", "medical",
    "healthcare", "health care", "centre", "center", "nursing home",
    "institute", "infirmary", "dispensary", "medicare", "medicals",
    "care", "life", "heart", "eye", "dental", "maternity", "trauma"
]

# Common noise words that appear in OCR and are NOT names
NOISE_WORDS = {
    "dear", "sir", "madam", "hello", "to", "from", "date", "the",
    "a", "an", "of", "in", "at", "on", "by", "for", "with", "and",
    "as", "is", "was", "are", "were", "ref", "no", "number", "id",
    "ward", "bed", "room", "floor", "page", "age", "sex", "gender",
    "male", "female", "mr", "mrs", "ms", "dr", "dr.", "prof"
}


# ============================================================
#  SECTION 2: TEXT CLEANING
#  Raw OCR text has a lot of garbage — normalize it first.
# ============================================================

def clean_ocr_text(raw_text: str) -> str:
    """
    Clean and normalize raw OCR text before extraction.

    Steps:
    - Normalize line endings
    - Remove non-printable characters
    - Collapse multiple spaces/blank lines
    - Preserve structure (colons, newlines) for label detection
    """
    if not raw_text:
        return ""

    text = raw_text

    # Normalize Windows/Mac line endings to Unix
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove non-printable ASCII characters (but keep \n and \t)
    text = re.sub(r'[^\x09\x0A\x20-\x7E\u00C0-\u024F\u0900-\u097F]', ' ', text)

    # Normalize multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse more than 2 blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ============================================================
#  SECTION 3: FUZZY MATCHING HELPER
#  OCR errors turn "Patient" into "Patlent" or "Pat1ent".
#  We use SequenceMatcher to catch these variations.
# ============================================================

def fuzzy_match_score(a: str, b: str) -> float:
    """
    Return similarity ratio between two strings (0.0 to 1.0).
    1.0 = identical, 0.0 = completely different.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_label_match(line: str, labels: list, threshold: float = 0.75) -> tuple:
    """
    Check if a line contains any label from the list — using fuzzy matching.

    Returns (matched_label, score) or (None, 0.0) if no match found.

    Example:
        line   = "Patlent Name: John Doe"   ← OCR error in "Patient"
        labels = ["patient name", ...]
        → returns ("patient name", 0.88)     ← still matches!
    """
    line_lower = line.lower()

    best_label = None
    best_score = 0.0

    for label in labels:
        # STRATEGY 1: Direct substring check (fast path)
        if label in line_lower:
            return (label, 1.0)

        # STRATEGY 2: Check each word segment the same length as label
        label_len = len(label)
        for i in range(len(line_lower) - label_len + 1):
            segment = line_lower[i:i + label_len]
            score = fuzzy_match_score(segment, label)
            if score > best_score:
                best_score = score
                best_label = label

    if best_score >= threshold:
        return (best_label, best_score)

    return (None, 0.0)


# ============================================================
#  SECTION 4: VALUE EXTRACTION AFTER A LABEL
#  Once we find a label line, extract the value that follows it.
#  Value can be on the same line (after a colon) or the next line.
# ============================================================

def extract_value_after_label(line: str, next_line: str = "") -> str:
    """
    Given a line containing a label, extract the value.

    Handles patterns like:
      "Patient Name: John Doe"          → "John Doe"
      "Patient Name"                    → check next_line
      "Patient Name - John Doe"         → "John Doe"
      "Patient Name ........ John Doe"  → "John Doe"
    """
    # Remove label separators: colon, dash, dots, pipes, equals
    # Pattern: everything after the first :, -, |, =, or ....
    value = re.sub(r'^[^:\-|=.]*[:|\-|=]', '', line).strip()

    # Also handle dot leaders (.....) used in forms
    value = re.sub(r'^\.+', '', value).strip()

    # If value is empty or too short after stripping, use next line
    if len(value) < 2 and next_line:
        value = next_line.strip()

    # Remove any trailing punctuation or noise
    value = re.sub(r'[,;.\s]+$', '', value).strip()

    return value


# ============================================================
#  SECTION 5: PATIENT NAME EXTRACTOR
# ============================================================

def extract_patient_name(text: str, lines: list) -> dict:
    """
    Extract patient name from OCR text.

    Strategies (tried in order, best confidence wins):
    1. Label-based: find "Patient Name:" and grab value after it
    2. Regex: look for "Mr./Mrs./Ms./Dr." followed by capitalized words
    3. Salutation pattern: lines starting with honorifics
    """
    candidates = []  # list of (value, confidence, method)

    # ── Strategy 1: Label-based extraction ───────────────────────────────────
    for i, line in enumerate(lines):
        matched_label, score = find_best_label_match(line, PATIENT_LABELS)
        if matched_label:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            value = extract_value_after_label(line, next_line)
            if value and len(value) > 1:
                # Filter out noise values
                if value.lower() not in NOISE_WORDS and not value.isdigit():
                    # Skip if value looks like a hospital name
                    val_lower = value.lower()
                    if any(suf in val_lower for suf in HOSPITAL_SUFFIXES):
                        continue
                    confidence = 0.7 + (score * 0.25)  # 0.70 – 0.95
                    candidates.append((clean_name(value), confidence, "label"))

    # ── Strategy 2: Honorific/salutation regex ────────────────────────────────
    # Matches: "Mr. John Doe", "Mrs. Jane Smith", "Dr. Anil Kumar"
    honorific_pattern = re.compile(
        r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        re.IGNORECASE
    )
    for match in honorific_pattern.finditer(text):
        full_name = match.group(0).strip()
        name_only = match.group(2).strip()
        if is_valid_name(name_only):
            candidates.append((clean_name(full_name), 0.80, "honorific"))

    # ── Strategy 3: ALL CAPS name lines (common in discharge summaries) ───────
    for line in lines:
        # Lines that are entirely uppercase words, 2-4 words, likely a name
        if re.match(r'^[A-Z][A-Z\s]{4,40}$', line.strip()):
            words = line.strip().split()
            if 2 <= len(words) <= 4 and all(len(w) > 1 for w in words):
                name = line.strip().title()
                if is_valid_name(name):
                    candidates.append((clean_name(name), 0.60, "allcaps"))

    return select_best_candidate(candidates, "patient_name")


# ============================================================
#  SECTION 6: HOSPITAL NAME EXTRACTOR
# ============================================================

def extract_hospital_name(text: str, lines: list) -> dict:
    """
    Extract hospital/clinic name.

    Strategies:
    1. Label-based: find "Hospital Name:" and grab value
    2. Suffix-based: find lines containing hospital/clinic keywords
    3. Header detection: first non-empty lines often contain hospital name
    """
    candidates = []

    # ── Strategy 1: Label-based ───────────────────────────────────────────────
    for i, line in enumerate(lines):
        matched_label, score = find_best_label_match(line, HOSPITAL_LABELS)
        if matched_label:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            value = extract_value_after_label(line, next_line)
            if value and len(value) > 2:
                candidates.append((value.strip(), 0.70 + (score * 0.25), "label"))

    # ── Strategy 2: Suffix-based line scan ────────────────────────────────────
    for line in lines:
        line_lower = line.lower()
        for suffix in HOSPITAL_SUFFIXES:
            if suffix in line_lower:
                # Check it's not just a label line (e.g. "Hospital Name:")
                if ':' not in line and len(line.strip()) > len(suffix) + 2:
                    # Boost confidence if multiple hospital-related words found
                    count = sum(1 for s in HOSPITAL_SUFFIXES if s in line_lower)
                    conf = min(0.55 + (count * 0.08), 0.85)
                    candidates.append((line.strip(), conf, "suffix"))
                    break  # one match per line is enough

    # ── Strategy 3: Header detection (first few non-empty lines) ─────────────
    non_empty = [l.strip() for l in lines if l.strip()]
    for line in non_empty[:5]:  # look at first 5 non-empty lines
        line_lower = line.lower()
        if any(s in line_lower for s in HOSPITAL_SUFFIXES):
            candidates.append((line.strip(), 0.65, "header"))

    return select_best_candidate(candidates, "hospital_name")


# ============================================================
#  SECTION 7: DISEASE / DIAGNOSIS EXTRACTOR
# ============================================================

def extract_disease(text: str, lines: list) -> dict:
    """
    Extract disease or diagnosis.

    Strategies:
    1. Label-based: find "Diagnosis:", "Disease:" etc.
    2. ICD code pattern: look for ICD-10 codes (e.g. J18.9) nearby text
    3. Medical keyword list: known common diagnoses
    """
    candidates = []

    # ── Strategy 1: Label-based ───────────────────────────────────────────────
    for i, line in enumerate(lines):
        matched_label, score = find_best_label_match(line, DISEASE_LABELS)
        if matched_label:
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            value = extract_value_after_label(line, next_line)
            if value and len(value) > 2 and not value.isdigit():
                candidates.append((value.strip(), 0.70 + (score * 0.25), "label"))

    # ── Strategy 2: ICD-10 code detection ────────────────────────────────────
    # ICD-10 format: letter + 2 digits + optional dot + digits (e.g. J18.9, A09, Z87.01)
    icd_pattern = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,4})?)\b')
    for match in icd_pattern.finditer(text):
        # Grab surrounding context (the line containing the ICD code)
        start = text.rfind('\n', 0, match.start()) + 1
        end = text.find('\n', match.end())
        context_line = text[start:end if end != -1 else len(text)].strip()
        # Remove the ICD code itself from the context to get disease name
        disease_text = re.sub(icd_pattern, '', context_line).strip()
        disease_text = re.sub(r'^[:\-\s]+', '', disease_text).strip()
        if disease_text and len(disease_text) > 2:
            candidates.append((disease_text, 0.85, "icd_code"))

    # ── Strategy 3: Common diagnosis keyword scan ─────────────────────────────
    # Common medical conditions — extend this list as needed
    COMMON_DIAGNOSES = [
        "diabetes", "hypertension", "fever", "malaria", "typhoid",
        "appendicitis", "fracture", "pneumonia", "tuberculosis", "tb",
        "dengue", "covid", "cancer", "carcinoma", "infection", "viral",
        "bacterial", "inflammatory", "cardiac", "renal", "failure",
        "anaemia", "anemia", "sepsis", "stroke", "epilepsy", "migraine",
        "gastritis", "hepatitis", "jaundice", "arthritis", "hernia",
        "cyst", "tumor", "tumour", "asthma", "copd", "diarrhoea",
        "diarrhea", "cholera", "uti", "urinary", "kidney", "liver",
        "heart", "blood pressure", "bp", "sugar", "thyroid"
    ]

    for line in lines:
        line_lower = line.lower()
        for diagnosis in COMMON_DIAGNOSES:
            if diagnosis in line_lower:
                # Clean: remove label part if present
                value = re.sub(r'.*(?:diagnosis|disease|condition)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
                if not value:
                    value = line.strip()
                if len(value) > 2:
                    candidates.append((value, 0.55, "keyword"))
                    break

    return select_best_candidate(candidates, "disease")


# ============================================================
#  SECTION 8: BILL AMOUNT EXTRACTOR
# ============================================================

def extract_bill_amount(text: str, lines: list) -> dict:
    """
    Extract total bill amount.

    Strategies:
    1. Label + amount regex: find "Total Amount: ₹45,000"
    2. Currency symbol scan: find standalone ₹ / Rs. / $ amounts
    3. Largest amount heuristic: in medical bills, total is usually the largest number
    """
    candidates = []

    # Currency/amount regex patterns
    # Handles: ₹45,000  |  Rs. 1,23,456.00  |  INR 50000  |  $2,500.50
    AMOUNT_PATTERN = re.compile(
        r'(?:₹|Rs\.?|INR|USD|\$)\s*(\d{1,3}(?:[,\s]\d{2,3})*(?:\.\d{1,2})?)'
        r'|(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?)'   # plain comma-formatted number
        r'|(\d{4,8}(?:\.\d{1,2})?)',                  # plain large number
        re.IGNORECASE
    )

    # ── Strategy 1: Label-based ───────────────────────────────────────────────
    for i, line in enumerate(lines):
        matched_label, score = find_best_label_match(line, AMOUNT_LABELS, threshold=0.70)
        if matched_label:
            # Search for amount pattern in this line AND next line
            search_text = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
            for match in AMOUNT_PATTERN.finditer(search_text):
                raw_amount = match.group(0)
                normalized = normalize_amount(raw_amount)
                if normalized:
                    conf = 0.70 + (score * 0.25)
                    # Extra boost for "total" or "grand total" labels
                    if "total" in line.lower() or "grand" in line.lower():
                        conf = min(conf + 0.10, 0.98)
                    candidates.append((normalized, conf, "label"))

    # ── Strategy 2: Currency symbol scan ─────────────────────────────────────
    for match in AMOUNT_PATTERN.finditer(text):
        raw_amount = match.group(0)
        # Only consider if it has a currency symbol or is large enough
        if any(sym in raw_amount for sym in ['₹', 'Rs', 'INR', '$', 'USD']):
            normalized = normalize_amount(raw_amount)
            if normalized:
                candidates.append((normalized, 0.60, "currency_symbol"))

    # ── Strategy 3: Largest amount heuristic ─────────────────────────────────
    # In medical bills, the grand total is usually the largest amount on the page
    all_amounts = []
    for match in AMOUNT_PATTERN.finditer(text):
        normalized = normalize_amount(match.group(0))
        if normalized:
            try:
                numeric_val = float(normalized.replace(',', ''))
                all_amounts.append((normalized, numeric_val))
            except:
                pass

    if all_amounts:
        # Sort by numeric value descending
        all_amounts.sort(key=lambda x: x[1], reverse=True)
        largest = all_amounts[0][0]
        # Low confidence — this is a fallback heuristic
        candidates.append((largest, 0.40, "largest_amount"))

    return select_best_candidate(candidates, "bill_amount")


# ============================================================
#  SECTION 9: HELPER UTILITIES
# ============================================================

def normalize_amount(raw: str) -> str:
    """
    Clean a raw amount string into a standard format.
    "Rs. 1,23,456.00" → "123456.00"
    "₹ 45,000"        → "45000"
    """
    if not raw:
        return ""
    # Remove currency symbols and labels
    cleaned = re.sub(r'[₹$]|Rs\.?|INR|USD', '', raw, flags=re.IGNORECASE)
    # Remove spaces
    cleaned = cleaned.replace(' ', '').strip()
    # Validate it looks like a number
    if re.match(r'^\d[\d,]*(?:\.\d{1,2})?$', cleaned):
        return cleaned
    return ""


def clean_name(name: str) -> str:
    """
    Clean a name string: remove extra spaces, fix capitalization.
    """
    if not name:
        return ""
    # Remove digits (OCR sometimes adds room/bed numbers)
    name = re.sub(r'\d+', '', name)
    # Remove common noise tokens
    name = re.sub(r'\b(?:age|sex|dob|ward|bed|room)\b', '', name, flags=re.IGNORECASE)
    # Normalize spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Title case
    return name.title()


def is_valid_name(name: str) -> bool:
    """
    Basic sanity check: does this look like a real person's name?
    - At least 2 characters
    - Not purely digits
    - Not a noise word
    - Contains only letters, spaces, dots, hyphens
    """
    if not name or len(name.strip()) < 2:
        return False
    if name.strip().isdigit():
        return False
    if name.strip().lower() in NOISE_WORDS:
        return False
    if not re.match(r'^[A-Za-z\s.\-\']+$', name.strip()):
        return False
    return True


def select_best_candidate(candidates: list, field_name: str) -> dict:
    """
    From a list of (value, confidence, method) tuples,
    select the best one and return structured result.

    Tie-breaking rules:
    1. Highest confidence score wins
    2. Among equal scores: label-based beats keyword-based beats heuristic
    """
    if not candidates:
        return {"value": "", "confidence": 0.0, "method": "none", "all_candidates": []}

    # Remove duplicates: if same value appears multiple times, keep highest confidence
    seen = {}
    for value, conf, method in candidates:
        key = value.lower().strip()
        if key not in seen or conf > seen[key][1]:
            seen[key] = (value, conf, method)

    unique_candidates = list(seen.values())

    # Sort by confidence descending
    METHOD_PRIORITY = {"label": 3, "icd_code": 2, "honorific": 2,
                       "suffix": 1, "header": 1, "keyword": 1,
                       "currency_symbol": 1, "allcaps": 0, "largest_amount": 0}

    unique_candidates.sort(
        key=lambda x: (x[1], METHOD_PRIORITY.get(x[2], 0)),
        reverse=True
    )

    best_value, best_conf, best_method = unique_candidates[0]

    return {
        "value": best_value,
        "confidence": round(best_conf, 2),
        "method": best_method,
        "all_candidates": [
            {"value": v, "confidence": round(c, 2), "method": m}
            for v, c, m in unique_candidates[:5]  # show top 5
        ]
    }


# ============================================================
#  SECTION 10: MAIN EXTRACTION PIPELINE
#  This is the entry point — call this with raw OCR text.
# ============================================================

def extract_claim_fields(raw_ocr_text: str) -> dict:
    """
    Main pipeline: take raw OCR text, extract all claim fields.

    Returns:
    {
        "patient_name":   "",
        "hospital_name":  "",
        "disease":        "",
        "bill_amount":    "",
        "confidence_scores": { ... },
        "extraction_details": { ... }   ← full candidate lists for debugging
    }
    """

    # Step 1: Clean the raw OCR text
    text = clean_ocr_text(raw_ocr_text)

    # Step 2: Split into lines for line-by-line processing
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Step 3: Extract each field independently
    patient_result  = extract_patient_name(text, lines)
    hospital_result = extract_hospital_name(text, lines)
    disease_result  = extract_disease(text, lines)
    amount_result   = extract_bill_amount(text, lines)

    # Step 4: Build the structured response
    result = {
        "patient_name":  patient_result["value"],
        "hospital_name": hospital_result["value"],
        "disease":       disease_result["value"],
        "bill_amount":   amount_result["value"],
        "confidence_scores": {
            "patient_name":  patient_result["confidence"],
            "hospital_name": hospital_result["confidence"],
            "disease":       disease_result["confidence"],
            "bill_amount":   amount_result["confidence"]
        },
        # Full details including all candidates — useful for debugging and UI
        "extraction_details": {
            "patient_name":  patient_result,
            "hospital_name": hospital_result,
            "disease":       disease_result,
            "bill_amount":   amount_result
        }
    }

    return result


# ============================================================
#  SECTION 11: STANDALONE TEST
#  Run this file directly to test with a sample OCR text.
# ============================================================

if __name__ == "__main__":
    # Sample OCR text (intentionally noisy to simulate real OCR output)
    SAMPLE_OCR = """
    APOLLO HOSPITALS LIMITED
    Apollo Hospital, Greams Road, Chennai - 600 006
    GSTIN: 33AAACA1234F1Z5

    DISCHARGE SUMMARY

    Patlent Name  : Mr. Rajesh Kumar
    Age / Sex     : 45 Years / Male
    IP Number     : IPD/2024/04521
    Ward          : General Ward  Bed No: 23
    Date of Adm.  : 12/03/2024
    Date of Dis.  : 18/03/2024

    Diagnoiss     : Typhoid Fever with Complications   ICD-10: A01.0

    BILL SUMMARY
    Room Charges                   Rs. 18,000
    Doctor Consultation            Rs.  6,500
    Medicines & Pharmacy           Rs. 12,350
    Lab Investigations             Rs.  8,200
    Other Charges                  Rs.  3,100
    ─────────────────────────────────────────
    Grand Total                    Rs. 48,150
    """

    result = extract_claim_fields(SAMPLE_OCR)

    print("\n" + "="*60)
    print("EXTRACTION RESULT")
    print("="*60)
    print(f"  Patient Name  : {result['patient_name']}")
    print(f"  Hospital Name : {result['hospital_name']}")
    print(f"  Disease       : {result['disease']}")
    print(f"  Bill Amount   : Rs. {result['bill_amount']}")
    print("\nConfidence Scores:")
    for field, score in result['confidence_scores'].items():
        bar = "█" * int(score * 20)
        print(f"  {field:<15}: {score:.0%}  {bar}")