# ============================================================
#  MediSuite-AI-Agent -- ai/text_cleaner.py
#  Feature 9: OCR text cleaning before AI summarization
# ============================================================

import re


def clean_ocr_text(raw_text: str) -> str:
    """
    Clean raw OCR text for summarization.
    Removes noise, page numbers, non-printable chars.
    Preserves medically relevant content.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\x09\x0A\x20-\x7E\u00C0-\u024F\u0900-\u097F]', ' ', text)

    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.match(r'^\d{1,4}$', s):   # page numbers
            continue
        if len(s) < 3:                   # noise
            continue
        clean_lines.append(s)

    text = ' '.join(clean_lines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'[-_=]{3,}', ' ', text)

    return text.strip()


def truncate_for_model(text: str, max_words: int = 700) -> str:
    """Truncate to max_words to fit transformer token limits safely."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compression_ratio(original: str, summary: str) -> float:
    orig = word_count(original)
    summ = word_count(summary)
    if orig == 0:
        return 0.0
    return round(summ / orig, 3)


def extract_key_sections(text: str) -> dict:
    """
    Extract key medical sections using regex patterns.
    Returns dict of section_name -> content.
    Best-effort — not all sections will be found in every document.
    """
    sections = {}
    PATTERNS = {
        "diagnosis": r'(?:diagnosis|diagnosed|condition|disease)\s*[:\-]?\s*([^.\n]{5,120})',
        "treatment": r'(?:treatment|treated|therapy|given|administered)\s*[:\-]?\s*([^.\n]{5,150})',
        "medicines": r'(?:medicine|medication|prescribed|tablet|capsule)\s*[:\-]?\s*([^.\n]{5,150})',
        "tests":     r'(?:test|investigation|lab|blood|scan|x-?ray)\s*[:\-]?\s*([^.\n]{5,150})',
        "followup":  r'(?:follow.?up|advice|instructions|discharge advice)\s*[:\-]?\s*([^.\n]{5,150})',
    }
    for section, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sections[section] = match.group(1).strip().title()
    return sections