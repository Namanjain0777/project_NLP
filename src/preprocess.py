"""
src/preprocess.py
-----------------
Per-language preprocessing pipeline for the multilingual SMS spam classifier.

Languages supported:
  - English  : spaCy en_core_web_sm  (tokenise → stopwords → lemmatise)
  - Hindi    : indic-nlp-library      (normalise → tokenise → stopwords)
  - Bengali  : indic-nlp-library      (normalise → char-level tokenise)
  - Assamese : Unicode NFC + char-level tokenise
               NOTE: IndicNLP has limited Assamese support — no stopword list
               and limited morphological tools available. Character-level
               tokenisation + NFC normalisation is used as a pragmatic fallback.

Common steps (all languages):
  - Remove URLs (http/https/www)
  - Remove phone numbers (10–13 digit sequences)
  - Strip excessive whitespace
  - Retain currency symbols (€ $ ¥ £ ₹ ₺) — strong spam signal

Feature engineering (all languages):
  - word_count, char_count
  - contains_currency_symbol, contains_number, contains_url, contains_phone_number
  - caps_ratio
  - script_type (latin / devanagari / bengali / assamese)
  - is_augmented  (passed through unchanged — set during augmentation)
"""

import re
import unicodedata
from typing import Optional

import pandas as pd

# ── Optional heavy imports — loaded lazily to avoid startup errors ──────────────
_spacy_nlp = None
_indic_normaliser = {}
_indic_tokeniser = {}
_indic_stopwords = {}


# ─────────────────────────────────────────────────────────────────────────────
# Script detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_script(text: str) -> str:
    """Determine the dominant Unicode script block of a text string.

    Returns one of: 'latin', 'devanagari', 'bengali', 'assamese', 'unknown'.
    Assamese and Bengali share the Bengali script block; we distinguish them
    via the language_detected metadata rather than script alone.
    """
    counts = {"latin": 0, "devanagari": 0, "bengali_or_assamese": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0000 <= cp <= 0x024F:          # Basic Latin + Latin Extended
            counts["latin"] += 1
        elif 0x0900 <= cp <= 0x097F:        # Devanagari
            counts["devanagari"] += 1
        elif 0x0980 <= cp <= 0x09FF:        # Bengali (shared with Assamese)
            counts["bengali_or_assamese"] += 1
        else:
            counts["other"] += 1

    dominant = max(counts, key=counts.get)
    if dominant == "latin":
        return "latin"
    elif dominant == "devanagari":
        return "devanagari"
    elif dominant == "bengali_or_assamese":
        # Caller should refine based on language_detected
        return "bengali_or_assamese"
    else:
        return "unknown"


def resolve_script_type(text: str, language: str) -> str:
    """Return a human-readable script_type label given text and detected language."""
    raw = detect_script(text)
    if raw == "bengali_or_assamese":
        return "assamese" if language in ("as", "assamese") else "bengali"
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns (compiled once at module load)
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PHONE_RE       = re.compile(r"\b\d{10,13}\b")
_WHITESPACE_RE  = re.compile(r"\s+")
_CURRENCY_RE    = re.compile(r"[€$¥£₹₺]")
_NUMBER_RE      = re.compile(r"\d")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def _remove_phone_numbers(text: str) -> str:
    return _PHONE_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Language-specific preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_english(text: str) -> str:
    """spaCy en_core_web_sm: tokenise → stopword removal → lemmatise."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            # Graceful degradation: return cleaned text without lemmatisation
            import warnings
            warnings.warn(f"spaCy model not available: {e}. Falling back to basic cleaning.")
            return text.lower()

    doc = _spacy_nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)


def _preprocess_hindi(text: str) -> str:
    """indic-nlp-library: normalise → tokenise → stopword removal."""
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.resources import word_list

        # Normalise
        factory = IndicNormalizerFactory()
        normaliser = factory.get_normalizer("hi")
        text = normaliser.normalize(text)

        # Tokenise
        tokens = indic_tokenize.trivial_tokenize(text, "hi")

        # Stopword removal (IndicNLP provides Hindi stopwords)
        try:
            stopwords = set(word_list.get_word_list("hi"))
        except Exception:
            stopwords = set()

        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        return " ".join(tokens)

    except ImportError as e:
        import warnings
        warnings.warn(f"indic-nlp-library not available: {e}. Using NFC normalisation only.")
        return unicodedata.normalize("NFC", text)


def _preprocess_bengali(text: str) -> str:
    """indic-nlp-library: normalise + character-level tokenisation.

    NOTE: Bengali morphological tools are limited; we use character-level
    tokenisation as the primary strategy rather than word-level lemmatisation.
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize

        factory = IndicNormalizerFactory()
        normaliser = factory.get_normalizer("bn")
        text = normaliser.normalize(text)

        # Word-level tokenise (character-level is implicit in model tokeniser)
        tokens = indic_tokenize.trivial_tokenize(text, "bn")
        return " ".join(tokens)

    except ImportError as e:
        import warnings
        warnings.warn(f"indic-nlp-library not available: {e}. Using NFC normalisation only.")
        return unicodedata.normalize("NFC", text)


def _preprocess_assamese(text: str) -> str:
    """Unicode NFC normalisation + character-level tokenisation.

    NOTE: IndicNLP has no dedicated Assamese stopword list or morphological
    tools as of 2024. We apply NFC normalisation and rely on the transformer
    (MuRIL) to handle Assamese script directly via its subword vocabulary.
    This limitation is documented here and in the project README.
    """
    # NFC normalisation — critical for Assamese Unicode consistency
    text = unicodedata.normalize("NFC", text)
    # No further tokenisation — MuRIL handles Assamese subwords natively
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Common cleaning (applied BEFORE language-specific steps)
# ─────────────────────────────────────────────────────────────────────────────

def _common_clean(text: str) -> str:
    """Remove URLs and phone numbers; normalise whitespace.
    Currency symbols are intentionally RETAINED as spam features.
    """
    text = _remove_urls(text)
    text = _remove_phone_numbers(text)
    text = _normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str, language: str) -> str:
    """Clean and normalise text according to the detected language.

    Parameters
    ----------
    text     : Raw SMS string
    language : ISO 639-1/3 code or name. Accepted values:
               'en' | 'english'
               'hi' | 'hindi'
               'bn' | 'bengali'
               'as' | 'assamese'
               Anything else falls back to basic NFC normalisation.

    Returns
    -------
    str : Preprocessed text ready for feature extraction or model tokenisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 — common cleaning (all languages)
    text = _common_clean(text)

    # Step 2 — language-specific processing
    lang = language.lower().strip()
    if lang in ("en", "english"):
        return _preprocess_english(text)
    elif lang in ("hi", "hindi"):
        return _preprocess_hindi(text)
    elif lang in ("bn", "bengali"):
        return _preprocess_bengali(text)
    elif lang in ("as", "assamese"):
        return _preprocess_assamese(text)
    else:
        # Unknown language — NFC + whitespace normalisation only
        return unicodedata.normalize("NFC", text)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a DataFrame that contains at minimum:
        - 'text'           : raw SMS string
        - 'language'       : detected language code
        - 'is_augmented'   : boolean flag (True for synthetic Assamese SPAM)

    Returns the DataFrame with additional columns:
        word_count, char_count, contains_currency_symbol, contains_number,
        contains_url, contains_phone_number, caps_ratio,
        language_detected, script_type
    """
    df = df.copy()

    # Ensure required columns exist
    if "is_augmented" not in df.columns:
        df["is_augmented"] = False
    if "language" not in df.columns:
        df["language"] = "en"

    df["language_detected"] = df["language"]

    # Text-based features (computed on RAW text before preprocessing)
    df["word_count"]                = df["text"].apply(lambda x: len(str(x).split()))
    df["char_count"]                = df["text"].apply(lambda x: len(str(x)))
    df["contains_currency_symbol"]  = df["text"].apply(
        lambda x: int(bool(_CURRENCY_RE.search(str(x))))
    )
    df["contains_number"]           = df["text"].apply(
        lambda x: int(bool(_NUMBER_RE.search(str(x))))
    )
    df["contains_url"]              = df["text"].apply(
        lambda x: int(bool(_URL_RE.search(str(x))))
    )
    df["contains_phone_number"]     = df["text"].apply(
        lambda x: int(bool(_PHONE_RE.search(str(x))))
    )
    df["caps_ratio"]                = df["text"].apply(
        lambda x: round(_caps_ratio(str(x)), 4)
    )
    df["script_type"]               = df.apply(
        lambda row: resolve_script_type(str(row["text"]), str(row["language"])), axis=1
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Language detection wrapper
# ─────────────────────────────────────────────────────────────────────────────

_lingua_detector = None


def detect_language(text: str) -> str:
    """Detect language using lingua-language-detector.

    Returns ISO 639-1 code: 'en', 'hi', 'bn', 'as', or 'unknown'.
    Falls back to 'unknown' if lingua is not installed or detection fails.
    """
    global _lingua_detector
    if _lingua_detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            # NOTE: lingua does NOT support Assamese. We handle it via
            # Assamese-exclusive Unicode codepoints (see below).
            _lingua_detector = (
                LanguageDetectorBuilder
                .from_languages(
                    Language.ENGLISH,
                    Language.HINDI,
                    Language.BENGALI,
                )
                .with_preloaded_language_models()
                .build()
            )
        except Exception:
            import warnings
            warnings.warn("lingua-language-detector unavailable. Using script detection.")
            _lingua_detector = "unavailable"

    # Assamese-exclusive codepoints within the Bengali Unicode block:
    #   U+09F0 = Assamese RA   U+09F1 = Assamese WA
    # These do not appear in standard Bengali text.
    _ASSAMESE_EXCLUSIVE = {chr(0x09F0), chr(0x09F1)}
    if any(ch in _ASSAMESE_EXCLUSIVE for ch in text):
        return "as"

    # Script-based fallback when lingua is unavailable
    script = detect_script(text)
    if _lingua_detector == "unavailable":
        _SCRIPT_MAP = {"devanagari": "hi", "latin": "en", "bengali_or_assamese": "bn"}
        return _SCRIPT_MAP.get(script, "unknown")

    # Lingua-based detection (EN / HI / BN)
    try:
        result = _lingua_detector.detect_language_of(text)
        if result is None:
            _SCRIPT_MAP = {"devanagari": "hi", "latin": "en", "bengali_or_assamese": "bn"}
            return _SCRIPT_MAP.get(script, "unknown")
        _LINGUA_TO_ISO = {"ENGLISH": "en", "HINDI": "hi", "BENGALI": "bn"}
        return _LINGUA_TO_ISO.get(result.name, "unknown")
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: process a full DataFrame end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def process_dataframe(df: pd.DataFrame, auto_detect_language: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline on a DataFrame.

    Expected input columns: 'text', 'label'
    Optional input columns: 'language' (if not present, auto-detect), 'is_augmented'

    Returns DataFrame with all engineered features and a 'text_clean' column
    containing the language-preprocessed text.
    """
    df = df.copy()

    if "is_augmented" not in df.columns:
        df["is_augmented"] = False

    if "language" not in df.columns or auto_detect_language:
        print("Auto-detecting languages via lingua...")
        df["language"] = df["text"].apply(detect_language)

    # Engineer features BEFORE cleaning (raw text → features)
    df = engineer_features(df)

    # Apply language-specific cleaning
    print("Applying per-language preprocessing...")
    df["text_clean"] = df.apply(
        lambda row: preprocess_text(str(row["text"]), str(row["language"])), axis=1
    )

    return df
