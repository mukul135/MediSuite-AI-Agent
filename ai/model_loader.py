# ============================================================
#  MediSuite-AI-Agent -- ai/model_loader.py
#  Feature 9: Singleton model loader — loads ONCE, caches forever
#
#  To change model: edit MODEL_NAME below, restart Flask.
#
#  Available models (ordered by size/quality):
#    sshleifer/distilbart-cnn-12-6   ~480MB  (recommended for CPU)
#    facebook/bart-large-cnn         ~1.6GB  (best quality, needs GPU or RAM)
#    t5-small                        ~240MB  (smallest, fastest)
#    t5-base                         ~850MB  (balanced)
# ============================================================

import os

# ── Change this to swap models ────────────────────────────────────────────────
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "summarization_model"
)

_pipeline     = None
_model_loaded = False
_load_error   = None


def get_summarization_pipeline():
    """
    Return cached pipeline. Downloads model on first call only.
    Raises RuntimeError with friendly message if unavailable.
    """
    global _pipeline, _model_loaded, _load_error

    if _model_loaded and _pipeline is not None:
        return _pipeline

    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        from transformers import pipeline as hf_pipeline
        import torch

        print(f"[MediSuite] Loading summarization model: {MODEL_NAME}")
        os.makedirs(CACHE_DIR, exist_ok=True)

        device = 0 if torch.cuda.is_available() else -1
        print(f"[MediSuite] Device: {'GPU' if device == 0 else 'CPU'}")

        _pipeline = hf_pipeline(
            "summarization",
            model=MODEL_NAME,
            cache_dir=CACHE_DIR,
            device=device,
        )
        _model_loaded = True
        print("[MediSuite] Summarization model loaded.")
        return _pipeline

    except ImportError as e:
        _load_error = f"Run: pip install transformers torch\nError: {e}"
        raise RuntimeError(_load_error)

    except Exception as e:
        _load_error = f"Model loading failed: {str(e)}"
        raise RuntimeError(_load_error)


def is_model_loaded() -> bool:
    return _model_loaded and _pipeline is not None


def get_model_name() -> str:
    return MODEL_NAME


def reset_model():
    """Force reload on next request (useful after changing MODEL_NAME)."""
    global _pipeline, _model_loaded, _load_error
    _pipeline = None
    _model_loaded = False
    _load_error = None