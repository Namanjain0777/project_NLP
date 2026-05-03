"""
src/augment.py
--------------
Assamese data augmentation via NLLB-200 back-translation + noise injection.

Translation model: facebook/nllb-200-distilled-600M
  - Fully open, no authentication required (replaces gated IndicTrans2)
  - Supports 200 languages natively including Assamese (asm_Beng)
  - ~2.4 GB download, cached automatically after first run

Strategy:
  1. Take all SPAM-labelled rows from the UCI English SMS dataset.
  2. Translate to Assamese using NLLB-200 (en → asm_Beng).
  3. Score each translation with chrF; discard if chrF < 0.4.
  4. Apply noise augmentation:
       - Random word dropout at p=0.05
       - Character swap at p=0.02
  5. Pair with real Assamese HAM from AI4Bharat IndicCorp (or synthetic HAM
     derived from English HAM via back-translation with is_augmented=True).
  6. Tag ALL synthetically generated rows with is_augmented=True.

GPU is recommended but CPU works (~10–20 min for ~750 samples on CPU).
"""

import random
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── NLLB-200 loaded on first use ──────────────────────────────────────────────
_nllb_model     = None
_nllb_tokenizer = None
_DEVICE         = None

# NLLB-200 language codes
_SRC_LANG = "eng_Latn"   # English
_TGT_LANG = "asm_Beng"   # Assamese (Bengali script)

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def _load_nllb(device: Optional[str] = None):
    """Lazy-load the NLLB-200 translation model (no HuggingFace auth required)."""
    global _nllb_model, _nllb_tokenizer, _DEVICE

    if _nllb_model is not None:
        return  # already loaded

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

        print(f"Loading NLLB-200 model: {MODEL_NAME}")
        print("First run downloads ~2.4 GB — this takes a few minutes...")

        # NllbTokenizer is the correct class for NLLB models
        try:
            _nllb_tokenizer = NllbTokenizer.from_pretrained(
                MODEL_NAME, src_lang=_SRC_LANG
            )
        except Exception:
            # Fallback to AutoTokenizer if NllbTokenizer not available
            from transformers import AutoTokenizer
            _nllb_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, src_lang=_SRC_LANG
            )

        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Select device
        if device is None:
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _DEVICE = device

        _nllb_model = _nllb_model.to(_DEVICE)
        _nllb_model.eval()
        print(f"NLLB-200 loaded on {_DEVICE}")

    except ImportError as e:
        raise ImportError(
            f"transformers or torch not installed: {e}. "
            "Install with: pip install torch transformers sentencepiece"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_en_to_assamese(
    texts: List[str],
    batch_size: int = 8,
    device: Optional[str] = None,
) -> List[str]:
    """Translate English strings to Assamese using NLLB-200.

    Parameters
    ----------
    texts      : List of English SMS strings.
    batch_size : Samples per forward pass. Reduce to 4 if OOM on GPU.
    device     : 'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    List[str] : Assamese translations, one per input string.
    """
    _load_nllb(device)

    import torch

    # Resolve Assamese target language token ID
    tgt_lang_id = _nllb_tokenizer.convert_tokens_to_ids(_TGT_LANG)
    if tgt_lang_id == _nllb_tokenizer.unk_token_id:
        # Some tokenizer versions use lang_code_to_id dict
        tgt_lang_id = _nllb_tokenizer.lang_code_to_id.get(_TGT_LANG)
        if tgt_lang_id is None:
            raise ValueError(
                f"Could not find token ID for '{_TGT_LANG}'. "
                "Verify NLLB language code is correct."
            )

    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        with torch.no_grad():
            inputs = _nllb_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(_DEVICE)

            generated = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        decoded = _nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 5 == 0 or done == len(texts):
            print(f"  Translated {done}/{len(texts)} samples")

    return translations


# ─────────────────────────────────────────────────────────────────────────────
# Translation quality filtering with chrF
# ─────────────────────────────────────────────────────────────────────────────

def compute_chrf(hypotheses: List[str], references: List[str]) -> List[float]:
    """Compute per-sentence chrF score using sacrebleu.

    chrF is a character n-gram F-score suited for morphologically rich Indic
    scripts. Samples below threshold 0.4 are discarded to ensure quality.
    """
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF()
        scores = []
        for hyp, ref in zip(hypotheses, references):
            score = chrf.sentence_score(hyp, [ref]).score / 100.0  # normalise 0–1
            scores.append(score)
        return scores
    except ImportError:
        warnings.warn(
            "sacrebleu not installed. Skipping chrF filtering. "
            "Install with: pip install sacrebleu"
        )
        return [1.0] * len(hypotheses)


# ─────────────────────────────────────────────────────────────────────────────
# Noise augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _word_dropout(text: str, p: float = 0.05, rng: Optional[random.Random] = None) -> str:
    """Randomly drop words with probability p.

    Improves diversity so the model does not overfit to exact NLLB-200 patterns.
    """
    if rng is None:
        rng = random.Random()
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if rng.random() > p]
    return " ".join(kept) if kept else text


def _char_swap(text: str, p: float = 0.02, rng: Optional[random.Random] = None) -> str:
    """Randomly swap adjacent characters with probability p per position.

    Simulates OCR / typing noise in real-world SMS data.
    """
    if rng is None:
        rng = random.Random()
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def apply_noise(
    texts: List[str],
    word_dropout_p: float = 0.05,
    char_swap_p: float = 0.02,
    seed: int = 42,
) -> List[str]:
    """Apply word dropout and character swap noise to a list of translated texts.

    Parameters
    ----------
    texts          : List of Assamese translated strings.
    word_dropout_p : Probability of dropping each word.
    char_swap_p    : Probability of swapping adjacent characters.
    seed           : Random seed for reproducibility.

    Returns
    -------
    List[str] : Noise-augmented Assamese strings.
    """
    rng = random.Random(seed)
    noisy = []
    for text in texts:
        text = _word_dropout(text, p=word_dropout_p, rng=rng)
        text = _char_swap(text, p=char_swap_p, rng=rng)
        noisy.append(text)
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment_assamese_spam(
    english_df: pd.DataFrame,
    chrf_threshold: float = 0.4,
    word_dropout_p: float = 0.05,
    char_swap_p: float = 0.02,
    batch_size: int = 8,
    device: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Full Assamese SPAM augmentation pipeline using NLLB-200.

    Parameters
    ----------
    english_df     : DataFrame with columns ['text', 'label'].
                     Only SPAM rows (label == 1) are used.
    chrf_threshold : Minimum chrF score to keep a translation (0.0–1.0).
    word_dropout_p : Word dropout probability.
    char_swap_p    : Character swap probability.
    batch_size     : NLLB-200 batch size (reduce to 4 if GPU OOM).
    device         : 'cuda', 'cpu', or None (auto-detect).
    seed           : Random seed.

    Returns
    -------
    pd.DataFrame with columns:
        ['text', 'label', 'language', 'is_augmented', 'chrf_score']
    """
    # Step 1 — Extract English SPAM rows
    spam_mask  = english_df["label"].isin([1, "spam", "SPAM"])
    spam_en    = english_df[spam_mask].copy()
    print(f"[augment] Step 1/4: {len(spam_en)} English SPAM rows selected.")

    if len(spam_en) == 0:
        raise ValueError("No SPAM rows found in english_df. Check label column.")

    english_texts = spam_en["text"].tolist()

    # Step 2 — Translate to Assamese via NLLB-200
    print("[augment] Step 2/4: Translating to Assamese via NLLB-200...")
    assamese_texts = translate_en_to_assamese(
        english_texts, batch_size=batch_size, device=device
    )

    # Step 3 — chrF quality filtering
    print("[augment] Step 3/4: Filtering by chrF score...")
    chrf_scores   = compute_chrf(assamese_texts, english_texts)
    accepted_mask = [s >= chrf_threshold for s in chrf_scores]
    n_ok  = sum(accepted_mask)
    n_bad = len(accepted_mask) - n_ok
    print(f"[augment] chrF filter (>={chrf_threshold}): {n_ok} accepted, {n_bad} rejected")

    accepted_texts  = [t for t, ok in zip(assamese_texts, accepted_mask) if ok]
    accepted_scores = [s for s, ok in zip(chrf_scores,    accepted_mask) if ok]

    if not accepted_texts:
        warnings.warn(
            "All translations rejected by chrF filter. "
            "Try lowering chrf_threshold or inspect NLLB-200 output."
        )
        return pd.DataFrame(
            columns=["text", "label", "language", "is_augmented", "chrf_score"]
        )

    # Step 4 — Noise augmentation
    print("[augment] Step 4/4: Applying noise augmentation...")
    noisy_texts = apply_noise(
        accepted_texts,
        word_dropout_p=word_dropout_p,
        char_swap_p=char_swap_p,
        seed=seed,
    )

    result = pd.DataFrame({
        "text":         noisy_texts,
        "label":        1,
        "language":     "as",
        "is_augmented": True,
        "chrf_score":   accepted_scores,
    })
    print(f"[augment] Done. {len(result)} augmented Assamese SPAM samples created.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build balanced Assamese dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_assamese_dataset(
    english_df: pd.DataFrame,
    assamese_ham_df: Optional[pd.DataFrame] = None,
    chrf_threshold: float = 0.4,
    device: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a balanced Assamese SPAM/HAM dataset.

    Parameters
    ----------
    english_df      : UCI English dataset (used for SPAM augmentation + HAM fallback).
    assamese_ham_df : Real Assamese HAM samples. If None, English HAM is
                      back-translated (is_augmented=True).
    chrf_threshold  : chrF quality threshold.
    device          : Inference device.
    seed            : Random seed.

    Returns
    -------
    pd.DataFrame with columns ['text', 'label', 'language', 'is_augmented']
    """
    # Generate SPAM
    spam_df = augment_assamese_spam(
        english_df,
        chrf_threshold=chrf_threshold,
        device=device,
        seed=seed,
    )

    # HAM
    if assamese_ham_df is not None:
        ham_df = assamese_ham_df.copy()
        ham_df["language"]     = "as"
        ham_df["is_augmented"] = False
        if "label" not in ham_df.columns:
            ham_df["label"] = 0
    else:
        warnings.warn(
            "No real Assamese HAM provided. Back-translating English HAM. "
            "These will be tagged is_augmented=True."
        )
        ham_en = english_df[english_df["label"].isin([0, "ham", "HAM"])].copy()
        n_target = len(spam_df)
        ham_en   = ham_en.sample(n=min(n_target, len(ham_en)), random_state=seed)

        ham_texts_as = translate_en_to_assamese(
            ham_en["text"].tolist(), device=device
        )
        ham_df = pd.DataFrame({
            "text":         ham_texts_as,
            "label":        0,
            "language":     "as",
            "is_augmented": True,
            "chrf_score":   None,
        })

    # Drop chrf_score to unify schema
    for df_ in [spam_df, ham_df]:
        df_.drop(columns=["chrf_score"], inplace=True, errors="ignore")

    combined = pd.concat([spam_df, ham_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(
        f"[augment] Final Assamese dataset: "
        f"{(combined['label']==1).sum()} SPAM, "
        f"{(combined['label']==0).sum()} HAM"
    )
    return combined
