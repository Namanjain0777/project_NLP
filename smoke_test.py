"""
smoke_test.py
-------------
Quick import and sanity check for the project.
Run from the project root:
    python smoke_test.py

This checks:
  1. All src modules import without error
  2. preprocess_text works for all 4 languages
  3. Feature engineering produces expected columns
  4. FastAPI app can be imported (endpoint definitions are valid)
  5. Dataset file exists

Does NOT require GPU or trained models.
"""

import sys
import io
import traceback
from pathlib import Path

# Force UTF-8 output so Unicode sample texts don't crash on Windows CP1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def section(title):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


def check(desc, fn):
    try:
        fn()
        print(f"{PASS} {desc}")
        return True
    except Exception as e:
        print(f"{FAIL} {desc}")
        print(f"       Error: {e}")
        return False


# -------------------------------------------------------------
section("1. Dataset")
# -------------------------------------------------------------

def _check_dataset():
    p = ROOT / "dataset" / "Spam SMS Collection"
    assert p.exists(), f"Dataset not found at {p}"
    import pandas as pd
    df = pd.read_csv(p, sep="\t", names=["label", "text"], encoding="latin-1")
    assert len(df) > 5000, f"Expected >5000 rows, got {len(df)}"

check("UCI SMS Spam Collection exists and is readable", _check_dataset)


# -------------------------------------------------------------
section("2. Core library imports")
# -------------------------------------------------------------

check("pandas", lambda: __import__("pandas"))
check("numpy",  lambda: __import__("numpy"))
check("sklearn", lambda: __import__("sklearn"))
check("fastapi", lambda: __import__("fastapi"))
check("pydantic", lambda: __import__("pydantic"))
check("streamlit", lambda: __import__("streamlit"))
check("requests", lambda: __import__("requests"))

# Heavy ML libs — warn rather than fail
def _check_torch():
    import torch
    print(f"         PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()}", end="")

def _check_transformers():
    import transformers
    print(f"         transformers {transformers.__version__}", end="")

def _check_lightgbm():
    import lightgbm
    print(f"         lightgbm {lightgbm.__version__}", end="")

for name, fn in [("torch", _check_torch), ("transformers", _check_transformers), ("lightgbm", _check_lightgbm)]:
    try:
        fn()
        print(f"\n{PASS} {name}")
    except ImportError:
        print(f"\n{WARN} {name} not installed — needed for training (notebooks 03–04)")


# -------------------------------------------------------------
section("3. src module imports")
# -------------------------------------------------------------

check("from src.preprocess import preprocess_text, engineer_features, detect_language",
      lambda: __import__("src.preprocess", fromlist=["preprocess_text", "engineer_features", "detect_language"]))

check("from src.augment import augment_assamese_spam, apply_noise",
      lambda: __import__("src.augment", fromlist=["augment_assamese_spam", "apply_noise"]))

check("from src.model import train_muril, train_fallback, predict, load_models",
      lambda: __import__("src.model", fromlist=["train_muril", "train_fallback", "predict", "load_models"]))

check("FastAPI app imports (src.api)",
      lambda: __import__("src.api", fromlist=["app"]))


# -------------------------------------------------------------
section("4. preprocess_text — per-language")
# -------------------------------------------------------------

from src.preprocess import preprocess_text

SAMPLES = {
    "en": "URGENT! You have won a FREE prize. Call 08001234567 now!",
    "hi": "बधाई हो! आपने 5 लाख रुपये जीते हैं। अभी कॉल करें।",
    "bn": "অভিনন্দন! আপনি ১০,০০০ টাকা পেয়েছেন।",
    "as": "অভিনন্দন! আপনি ৫০,০০০ টকা জিকিছে।",
}

for lang, text in SAMPLES.items():
    def _run(t=text, l=lang):
        result = preprocess_text(t, l)
        assert isinstance(result, str) and len(result) > 0, f"Empty result for {l}"
    check(f"preprocess_text('{lang}')", _run)


# -------------------------------------------------------------
section("5. engineer_features — column check")
# -------------------------------------------------------------

def _check_features():
    import pandas as pd
    from src.preprocess import engineer_features

    df = pd.DataFrame([
        {"text": "Win FREE cash now! Call 0800123456", "language": "en", "label": 1, "is_augmented": False},
        {"text": "See you at 5pm tomorrow", "language": "en", "label": 0, "is_augmented": False},
    ])
    df_out = engineer_features(df)

    required = [
        "word_count", "char_count", "contains_currency_symbol",
        "contains_number", "contains_url", "contains_phone_number",
        "caps_ratio", "language_detected", "script_type",
    ]
    missing = [c for c in required if c not in df_out.columns]
    assert not missing, f"Missing columns: {missing}"
    # spam row should have contains_number=1
    assert df_out.loc[0, "contains_number"] == 1

check("All 9 engineered feature columns present + values correct", _check_features)


# -------------------------------------------------------------
section("6. augment helpers")
# -------------------------------------------------------------

def _check_noise():
    from src.augment import apply_noise
    texts = ["Free prize! Win now!", "Call us at 08001234567"]
    noisy = apply_noise(texts, word_dropout_p=0.0, char_swap_p=0.0)  # 0 noise -> unchanged
    assert noisy == texts, "Zero-noise augmentation should return identical texts"

check("apply_noise with p=0 returns unchanged texts", _check_noise)


# -------------------------------------------------------------
section("7. FastAPI app validation")
# -------------------------------------------------------------

def _check_api_routes():
    from src.api import app
    routes = {r.path for r in app.routes}
    assert "/predict" in routes, "/predict route missing"
    assert "/health" in routes, "/health route missing"

check("FastAPI has /predict and /health routes", _check_api_routes)


# -------------------------------------------------------------
section("8. Project file structure")
# -------------------------------------------------------------

expected_files = [
    "requirements.txt",
    "README.md",
    ".gitignore",
    "run.bat",
    "src/__init__.py",
    "src/preprocess.py",
    "src/augment.py",
    "src/model.py",
    "src/api.py",
    "src/app.py",
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_augmentation.ipynb",
    "notebooks/04_model_training.ipynb",
    "notebooks/05_evaluation.ipynb",
    "data/.gitkeep",
    "models/.gitkeep",
]

for f in expected_files:
    path = ROOT / f
    def _file_check(p=path, name=f):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {name}")
    check(f"Exists: {f}", _file_check)

# -------------------------------------------------------------
print("\n" + "="*55)
print("  Smoke test complete.")
print("  Next steps:")
print("    1. pip install -r requirements.txt")
print("    2. python -m spacy download en_core_web_sm")
print("    3. Run notebooks in order: 01 -> 03 -> 02 -> 04 -> 05")
print("    4. Double-click run.bat (or see README for manual steps)")
print("="*55 + "\n")
