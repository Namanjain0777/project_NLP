"""
src/model.py
------------
Two-tier spam classification architecture:

Tier 1 — Multilingual Transformer (Primary)
  Backbone: google/muril-base-cased
  Why MuRIL over IndicBERT?
    - MuRIL was pre-trained on 17 Indian languages including Assamese.
    - IndicBERT does NOT support Assamese.
    - MuRIL natively supports English, Hindi, Bengali, and Assamese — all
      four target languages in this project.
  Training: HuggingFace Trainer API, weighted cross-entropy loss, AdamW,
            linear warmup, early stopping on validation macro F1.

Tier 2 — Classical ML Fallback (LightGBM)
  Activated when: transformer confidence < 0.75 OR lingua returns 'unknown'.
  Features: TF-IDF char n-grams (2–5) + engineered numeric features.
  Imbalance: SMOTE on TF-IDF feature space.
  Combines: TF-IDF vectors + [word_count, char_count, caps_ratio,
            contains_currency_symbol, contains_number, contains_url,
            contains_phone_number]
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR         = Path(__file__).parent.parent / "models"
MURIL_MODEL_NAME  = "google/muril-base-cased"
CONFIDENCE_THRESH = 0.75           # Below this → fall back to Tier 2
MAX_SEQ_LEN       = 128
BATCH_SIZE        = 32
MAX_EPOCHS        = 5
WARMUP_RATIO      = 0.10
PATIENCE          = 2              # Early stopping patience (macro F1)

FALLBACK_TFIDF_PATH = MODEL_DIR / "fallback_tfidf.pkl"
FALLBACK_LGBM_PATH  = MODEL_DIR / "fallback_lgbm.pkl"
MURIL_SAVE_DIR      = MODEL_DIR / "muril_finetuned"

ENGINEERED_FEATURES = [
    "word_count", "char_count", "caps_ratio",
    "contains_currency_symbol", "contains_number",
    "contains_url", "contains_phone_number",
]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: LightGBM Fallback
# ─────────────────────────────────────────────────────────────────────────────

def train_fallback(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    ngram_range: Tuple[int, int] = (2, 5),
    max_tfidf_features: int = 30_000,
) -> None:
    """Train and save the LightGBM fallback model.

    Parameters
    ----------
    df_train         : Training DataFrame with 'text_clean', 'label', and
                       engineered feature columns.
    df_val           : Optional validation DataFrame for early stopping.
    ngram_range      : Character n-gram range for TF-IDF.
    max_tfidf_features : Maximum TF-IDF vocabulary size.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        import lightgbm as lgb
        import scipy.sparse as sp
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install lightgbm, imbalanced-learn, scikit-learn."
        ) from e

    print("[fallback] Building TF-IDF features (char n-grams)...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_tfidf_features,
        sublinear_tf=True,
        strip_accents=None,     # preserve Indic scripts
    )

    X_text = tfidf.fit_transform(df_train["text_clean"].fillna(""))
    X_eng  = df_train[ENGINEERED_FEATURES].fillna(0).values
    X      = sp.hstack([X_text, sp.csr_matrix(X_eng)]).toarray()
    y      = df_train["label"].values

    # SMOTE on the combined feature space
    print("[fallback] Applying SMOTE for class balance...")
    smote  = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # LightGBM classifier
    print("[fallback] Training LightGBM classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        class_weight="balanced",
    )

    if df_val is not None:
        X_val_text = tfidf.transform(df_val["text_clean"].fillna(""))
        X_val_eng  = df_val[ENGINEERED_FEATURES].fillna(0).values
        X_val      = sp.hstack([X_val_text, sp.csr_matrix(X_val_eng)]).toarray()
        y_val      = df_val["label"].values
        clf.fit(
            X_res, y_res,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
    else:
        clf.fit(X_res, y_res)

    # Save artefacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FALLBACK_TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(FALLBACK_LGBM_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"[fallback] Saved TF-IDF → {FALLBACK_TFIDF_PATH}")
    print(f"[fallback] Saved LightGBM → {FALLBACK_LGBM_PATH}")


def predict_fallback(
    texts_clean: list,
    feature_rows: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the LightGBM fallback model.

    Returns (labels, confidences) as numpy arrays.
    labels: 0 = HAM, 1 = SPAM
    confidences: probability of the predicted class.
    """
    try:
        import scipy.sparse as sp
    except ImportError as e:
        raise ImportError("scipy required for fallback inference.") from e

    with open(FALLBACK_TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(FALLBACK_LGBM_PATH, "rb") as f:
        clf   = pickle.load(f)

    X_text = tfidf.transform(texts_clean)

    if feature_rows is not None:
        X_eng = feature_rows[ENGINEERED_FEATURES].fillna(0).values
        X = sp.hstack([X_text, sp.csr_matrix(X_eng)]).toarray()
    else:
        X = X_text.toarray()

    probas  = clf.predict_proba(X)          # shape (n, 2)
    labels  = np.argmax(probas, axis=1)
    confs   = probas[np.arange(len(labels)), labels]
    return labels, confs


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: MuRIL Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute inverse-frequency class weights for weighted cross-entropy."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


class _WeightedTrainer:
    """HuggingFace Trainer subclass with weighted cross-entropy loss.

    This avoids oversampling and instead handles class imbalance directly
    in the loss function — the recommended approach for transformer models.
    """
    def __new__(cls, class_weights: Dict[int, float], *args, **kwargs):
        # Import here to keep module loadable without transformers installed
        try:
            from transformers import Trainer
        except ImportError as e:
            raise ImportError("transformers library required for MuRIL training.") from e

        import torch
        import torch.nn as nn

        class WeightedTrainer(Trainer):
            def __init__(self_inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self_inner._weights = torch.tensor(
                    [class_weights[0], class_weights[1]], dtype=torch.float
                )

            def compute_loss(self_inner, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                device = logits.device
                loss_fn = nn.CrossEntropyLoss(
                    weight=self_inner._weights.to(device)
                )
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        return WeightedTrainer(*args, **kwargs)


def train_muril(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str = MURIL_MODEL_NAME,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    warmup_ratio: float = WARMUP_RATIO,
    patience: int = PATIENCE,
    output_dir: Optional[Path] = None,
) -> None:
    """Fine-tune MuRIL for binary spam classification.

    Parameters
    ----------
    df_train    : Training DataFrame with 'text' (raw) and 'label' columns.
    df_val      : Validation DataFrame (same schema).
    model_name  : HuggingFace model hub ID.
    max_seq_len : Tokeniser max length.
    batch_size  : Per-device training batch size.
    max_epochs  : Maximum number of training epochs.
    warmup_ratio: Fraction of total steps for linear warmup.
    patience    : Early stopping patience on validation macro F1.
    output_dir  : Where to save the fine-tuned model. Defaults to MODEL_DIR/muril_finetuned.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        from datasets import Dataset
        from sklearn.metrics import f1_score
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install transformers, datasets, torch."
        ) from e

    save_dir = output_dir or MURIL_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokeniser ──────────────────────────────────────────────────────────
    print(f"[muril] Loading tokeniser: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )

    # ── Datasets ───────────────────────────────────────────────────────────
    train_hf = Dataset.from_pandas(
        df_train[["text", "label"]].rename(columns={"label": "labels"})
    )
    val_hf   = Dataset.from_pandas(
        df_val[["text", "label"]].rename(columns={"label": "labels"})
    )

    train_hf = train_hf.map(tokenize_fn, batched=True)
    val_hf   = val_hf.map(tokenize_fn, batched=True)
    train_hf.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_hf.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])

    # ── Class weights ──────────────────────────────────────────────────────
    class_weights = _compute_class_weights(df_train["label"].values)
    print(f"[muril] Class weights: {class_weights}")

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"[muril] Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # ── Metrics ────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    # ── Training args ──────────────────────────────────────────────────────
    total_steps = (len(df_train) // batch_size) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=str(save_dir / "logs"),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",          # disable wandb / tensorboard by default
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = _WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    print("[muril] Starting training...")
    trainer.train()
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[muril] Model saved → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference — Two-tier prediction
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache for loaded models
_muril_pipeline = None
_fallback_loaded = False


def load_models() -> None:
    """Load both tiers into memory. Call once at application startup."""
    global _muril_pipeline, _fallback_loaded

    # Tier 1 — MuRIL
    try:
        from transformers import pipeline as hf_pipeline
        model_path = str(MURIL_SAVE_DIR)
        print(f"[inference] Loading MuRIL from {model_path}")
        _muril_pipeline = hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if _cuda_available() else -1,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_all_scores=True,
        )
        print("[inference] MuRIL loaded.")
    except Exception as e:
        warnings.warn(f"Could not load MuRIL: {e}. Will use fallback only.")
        _muril_pipeline = None

    # Tier 2 — LightGBM
    try:
        if FALLBACK_TFIDF_PATH.exists() and FALLBACK_LGBM_PATH.exists():
            # Pre-load by doing a dummy import (actual load happens in predict_fallback)
            _fallback_loaded = True
            print("[inference] LightGBM fallback available.")
        else:
            _fallback_loaded = False
            warnings.warn("Fallback model files not found. Run train_fallback() first.")
    except Exception as e:
        warnings.warn(f"Fallback check failed: {e}")
        _fallback_loaded = False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def predict(
    text: str,
    text_clean: str,
    language: str,
    feature_row: Optional[pd.Series] = None,
    confidence_threshold: float = CONFIDENCE_THRESH,
) -> Dict:
    """Two-tier prediction for a single SMS.

    Parameters
    ----------
    text               : Raw SMS text.
    text_clean         : Preprocessed (cleaned) text.
    language           : Detected language code ('en', 'hi', 'bn', 'as', 'unknown').
    feature_row        : pd.Series of engineered features (optional, for fallback).
    confidence_threshold : Below this → route to Tier 2 fallback.

    Returns
    -------
    dict with keys: label (str), confidence (float), tier (int), language_detected (str)
    """
    label_str_map = {0: "HAM", 1: "SPAM"}
    used_tier = 1

    # ── Tier 1: MuRIL ──────────────────────────────────────────────────────
    muril_label = None
    muril_conf  = 0.0

    if _muril_pipeline is not None and language != "unknown":
        try:
            result = _muril_pipeline(text)[0]
            # result is list of {label: 'LABEL_0'/'LABEL_1', score: float}
            scores = {int(r["label"].split("_")[1]): r["score"] for r in result}
            muril_label = max(scores, key=scores.get)
            muril_conf  = scores[muril_label]
        except Exception as e:
            warnings.warn(f"MuRIL inference failed: {e}. Using fallback.")

    # ── Route decision ──────────────────────────────────────────────────────
    if (
        muril_label is None
        or muril_conf < confidence_threshold
        or language == "unknown"
    ):
        # Route to Tier 2
        used_tier = 2
        if _fallback_loaded:
            feat_df = (
                pd.DataFrame([feature_row]) if feature_row is not None
                else None
            )
            labels, confs = predict_fallback([text_clean], feat_df)
            final_label = int(labels[0])
            final_conf  = float(confs[0])
        else:
            # Last resort: use MuRIL result even if low-confidence
            if muril_label is not None:
                final_label = muril_label
                final_conf  = muril_conf
                used_tier   = 1
            else:
                raise RuntimeError(
                    "Both MuRIL and fallback are unavailable. "
                    "Run train_muril() and train_fallback() first."
                )
    else:
        final_label = muril_label
        final_conf  = muril_conf

    return {
        "label":             label_str_map[final_label],
        "confidence":        round(final_conf, 4),
        "tier":              used_tier,
        "language_detected": language,
    }
