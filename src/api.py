"""
src/api.py
----------
FastAPI backend for the multilingual SMS spam classifier.

Endpoints:
  POST /predict  — classify an SMS, return label + confidence + language
  GET  /health   — model load status and version

Run locally:
  uvicorn src.api:app --reload --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Local modules ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import detect_language, preprocess_text, engineer_features
from src.model import load_models, predict as model_predict

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("spam_api")

# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle — load models once at startup
# ─────────────────────────────────────────────────────────────────────────────

MODEL_VERSION = "1.0.0"
_startup_ok   = False
_startup_error = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup; release resources at shutdown."""
    global _startup_ok, _startup_error
    logger.info("Loading models...")
    try:
        load_models()
        _startup_ok = True
        logger.info("Models loaded successfully.")
    except Exception as e:
        _startup_error = str(e)
        logger.error(f"Model load failed: {e}")
        # Don't crash the server — /health will report the failure
    yield
    logger.info("API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual SMS Spam Classifier",
    description=(
        "Classify SMS messages as SPAM or HAM across English, Hindi, "
        "Bengali, and Assamese using a two-tier MuRIL + LightGBM architecture."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Allow Streamlit frontend on localhost:8501 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {"auto", "en", "hi", "bn", "as",
                       "english", "hindi", "bengali", "assamese"}

_LANG_NORMALISE = {
    "english": "en", "hindi": "hi", "bengali": "bn", "assamese": "as",
}


class PredictRequest(BaseModel):
    sms: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The SMS message text to classify.",
        examples=["Congratulations! You've won a FREE ticket. Call now to claim!"],
    )
    language: Optional[str] = Field(
        default="auto",
        description=(
            "Language of the SMS. Use 'auto' for automatic detection. "
            "Accepted: auto, en, hi, bn, as (or full names)."
        ),
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        v = (v or "auto").lower().strip()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Accepted values: {sorted(SUPPORTED_LANGUAGES)}"
            )
        return _LANG_NORMALISE.get(v, v)


class PredictResponse(BaseModel):
    label:             str   = Field(..., description="'SPAM' or 'HAM'")
    confidence:        float = Field(..., description="Confidence score [0, 1]")
    language_detected: str   = Field(..., description="ISO 639-1 language code")
    tier:              int   = Field(..., description="1 = MuRIL, 2 = LightGBM fallback")
    latency_ms:        float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status:              str
    model_version:       str
    models_loaded:       bool
    startup_error:       Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — request logging
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed:.1f} ms)"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Return model load status and API version."""
    return HealthResponse(
        status="ok" if _startup_ok else "degraded",
        model_version=MODEL_VERSION,
        models_loaded=_startup_ok,
        startup_error=_startup_error if not _startup_ok else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(payload: PredictRequest):
    """Classify an SMS message as SPAM or HAM.

    - Language is auto-detected by default using lingua-language-detector.
    - The primary model is MuRIL (Tier 1); if confidence < 0.75 or language
      is unknown, the LightGBM fallback (Tier 2) is used.
    """
    t0 = time.perf_counter()

    try:
        # ── Language detection ─────────────────────────────────────────────
        lang = payload.language
        if lang == "auto":
            lang = detect_language(payload.sms)
            logger.debug(f"Auto-detected language: {lang}")

        # ── Preprocessing ─────────────────────────────────────────────────
        text_clean = preprocess_text(payload.sms, lang)

        # ── Engineered features (for fallback) ────────────────────────────
        import pandas as pd
        feat_df = engineer_features(
            pd.DataFrame([{"text": payload.sms, "language": lang, "is_augmented": False}])
        )
        feature_row = feat_df.iloc[0]

        # ── Two-tier inference ────────────────────────────────────────────
        result = model_predict(
            text=payload.sms,
            text_clean=text_clean,
            language=lang,
            feature_row=feature_row,
        )

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Prediction: {result['label']} (conf={result['confidence']:.3f}, "
            f"tier={result['tier']}, lang={result['language_detected']}, "
            f"latency={latency:.1f}ms)"
        )

        return PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
            language_detected=result["language_detected"],
            tier=result["tier"],
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
