# ============================================================
#  MediSuite-AI-Agent -- ai/medical_summarizer.py
#  Feature 9: Main Medical Report Summarization Engine
#
#  Strategy:
#  1. Try Hugging Face transformer (BART/T5)
#  2. Fall back to rule-based extractive summary if model unavailable
#  Never crashes Flask — all errors are caught and returned.
#
#  DO NOT import claim_extractor, medical_nlp, claim_verifier,
#  predict_claim, or detect_fraud from here.
# ============================================================

import re
from datetime import datetime

from ai.text_cleaner import (
    clean_ocr_text, truncate_for_model,
    word_count, compression_ratio, extract_key_sections
)
from ai.model_loader import get_summarization_pipeline, get_model_name


# ── Summary length configurations ─────────────────────────────────────────────
SUMMARY_CONFIGS = {
    "short":    {"min_length": 30,  "max_length": 80,  "label": "Short"},
    "medium":   {"min_length": 80,  "max_length": 180, "label": "Medium"},
    "detailed": {"min_length": 150, "max_length": 350, "label": "Detailed"},
}

MEDICAL_KEYWORDS = [
    "diagnosis", "diagnosed", "disease", "condition", "treatment",
    "medicine", "medication", "prescribed", "admitted", "discharged",
    "fever", "pain", "test", "blood", "platelet", "surgery",
    "doctor", "hospital", "patient", "follow", "advice",
    "improved", "stable", "critical", "infection", "report"
]


# ============================================================
#  SECTION 1: RULE-BASED FALLBACK (no transformer needed)
# ============================================================

def _rule_based_summary(text: str, summary_type: str) -> str:
    """
    Extractive summary by sentence scoring.
    Used when transformer model is unavailable (no internet / CPU only).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text[:300] + "..." if len(text) > 300 else text

    def score(sent):
        sent_lower = sent.lower()
        return sum(1 for kw in MEDICAL_KEYWORDS if kw in sent_lower)

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    n = {"short": 2, "medium": 4, "detailed": 7}.get(summary_type, 3)
    top_idx = sorted([i for i, _ in scored[:n]])
    selected = [sentences[i] for i in top_idx if i < len(sentences)]

    return ' '.join(selected) if selected else sentences[0]


# ============================================================
#  SECTION 2: TRANSFORMER SUMMARIZER
# ============================================================

def _transformer_summary(text: str, summary_type: str) -> str:
    """Generate AI summary using loaded Hugging Face pipeline."""
    cfg      = SUMMARY_CONFIGS.get(summary_type, SUMMARY_CONFIGS["medium"])
    pipe     = get_summarization_pipeline()
    safe_txt = truncate_for_model(text, max_words=700)

    result = pipe(
        safe_txt,
        min_length=cfg["min_length"],
        max_length=cfg["max_length"],
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"].strip()


# ============================================================
#  SECTION 3: RESULT BUILDER
# ============================================================

def _build_result(original: str, summary: str, summary_type: str,
                  model_used: str, ocr_result_id=None, claim_id=None) -> dict:
    return {
        "summary_type":        SUMMARY_CONFIGS.get(summary_type, {}).get("label", summary_type.title()),
        "summary":             summary,
        "key_sections":        extract_key_sections(original),
        "word_count":          word_count(summary),
        "original_word_count": word_count(original),
        "compression_ratio":   compression_ratio(original, summary),
        "model_used":          model_used,
        "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ocr_result_id":       ocr_result_id,
        "claim_id":            claim_id,
        "error":               None,
    }


def _error_result(msg: str, summary_type: str,
                  ocr_result_id=None, claim_id=None) -> dict:
    return {
        "summary_type": summary_type.title(),
        "summary": "", "key_sections": {},
        "word_count": 0, "original_word_count": 0,
        "compression_ratio": 0.0, "model_used": "none",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ocr_result_id": ocr_result_id, "claim_id": claim_id,
        "error": msg,
    }


# ============================================================
#  SECTION 4: MAIN PIPELINE
# ============================================================

def summarize(ocr_text: str,
              summary_type: str = "medium",
              ocr_result_id=None,
              claim_id=None) -> dict:
    """
    Main entry point.

    Args:
        ocr_text:      Raw OCR text from Feature 3
        summary_type:  "short" | "medium" | "detailed"
        ocr_result_id: FK to ocr_results table (optional)
        claim_id:      FK to claims table (optional)

    Returns:
        Structured dict — never raises an exception.
    """
    if not ocr_text or not ocr_text.strip():
        return _error_result(
            "No OCR text provided. Extract text from a medical report first.",
            summary_type, ocr_result_id, claim_id
        )

    if summary_type not in SUMMARY_CONFIGS:
        summary_type = "medium"

    cleaned = clean_ocr_text(ocr_text)

    if word_count(cleaned) < 10:
        return _error_result(
            "Text too short to summarize meaningfully.",
            summary_type, ocr_result_id, claim_id
        )

    # Try transformer → fall back to rule-based
    try:
        summary    = _transformer_summary(cleaned, summary_type)
        model_used = get_model_name()
    except Exception as e:
        print(f"[MediSuite] Transformer unavailable ({e}), using rule-based fallback.")
        summary    = _rule_based_summary(cleaned, summary_type)
        model_used = "Rule-Based Extractive (fallback)"

    return _build_result(cleaned, summary, summary_type, model_used,
                         ocr_result_id, claim_id)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE = """
    Patient Name: Mr. Rajesh Kumar. Admitted: 10/03/2024. Discharged: 17/03/2024.
    Consulting: Dr. Priya Nair (Cardiologist).
    Diagnosis: Dengue Fever with Thrombocytopenia, Type 2 Diabetes Mellitus.
    Patient presented with high-grade fever for five days, severe headache, body ache,
    vomiting, and rash. Platelet count severely reduced to 72,000 per microliter.
    Investigations: CBC, Platelet Count, Dengue NS1 Antigen Positive, LFT, KFT, HbA1c 8.2%.
    Treatment: IV Fluids Normal Saline and DNS. Paracetamol 500mg TDS. Azithromycin 500mg OD.
    Insulin Lantus 10 units at bedtime. Blood Transfusion 1 unit on day 3.
    Patient improved from day 3. Platelet count recovered to 145000 by discharge. Condition stable.
    Follow-up: Review after 1 week. Monitor platelet count weekly. Avoid NSAIDs.
    """
    for t in ["short","medium","detailed"]:
        r = summarize(SAMPLE, summary_type=t)
        print(f"\n[{r['summary_type']}] {r['original_word_count']}w → {r['word_count']}w | {r['model_used']}")
        print(r['summary'])