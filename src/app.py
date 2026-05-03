"""
src/app.py
----------
Streamlit frontend for the multilingual SMS spam classifier.

Run locally:
  streamlit run src/app.py

Requires the FastAPI backend to be running at http://localhost:8000
  uvicorn src.api:app --reload --port 8000
"""

import sys
import time
from pathlib import Path

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

LANG_OPTIONS = {
    "🔍 Auto-detect": "auto",
    "🇬🇧 English":    "en",
    "🇮🇳 Hindi":      "hi",
    "🇧🇩 Bengali":    "bn",
    "🏔️ Assamese":    "as",
}

# Example SMS messages: (language_key, label, text)
EXAMPLES = [
    # English SPAM
    ("🇬🇧 English", "SPAM",
     "URGENT: You've won a £1,000 Tesco gift card! Call 08081570066 to claim NOW. Offer expires today!"),
    ("🇬🇧 English", "SPAM",
     "FREE entry: Win FA Cup Final tickets! To apply, send TEXT FA to 81872. 50p/msg. T&Cs at www.facomp.com"),
    # English HAM
    ("🇬🇧 English", "HAM",
     "Hey, are you coming to the party tonight? Let me know by 6pm!"),

    # Hindi SPAM
    ("🇮🇳 Hindi", "SPAM",
     "बधाई हो! आपने 5,00,000 रुपये जीते हैं। अभी क्लेम करने के लिए 9876543210 पर कॉल करें।"),
    ("🇮🇳 Hindi", "SPAM",
     "मुफ्त में iPhone जीतें! अभी रजिस्टर करें: www.offer-india.com पर जाएं। सीमित समय का ऑफर!"),
    # Hindi HAM
    ("🇮🇳 Hindi", "HAM",
     "कल मीटिंग 10 बजे है। क्या आप आ सकते हैं?"),

    # Bengali SPAM
    ("🇧🇩 Bengali", "SPAM",
     "অভিনন্দন! আপনি ১০,০০০ টাকা পেয়েছেন। এখনই দাবি করুন: 01812345678 নম্বরে কল করুন।"),
    ("🇧🇩 Bengali", "SPAM",
     "বিনামূল্যে স্মার্টফোন জিতুন! নিবন্ধন করুন: www.bangla-offer.com"),
    # Bengali HAM
    ("🇧🇩 Bengali", "HAM",
     "আজ বিকালে আসবে? চা খাওয়া যাবে।"),

    # Assamese SPAM
    ("🏔️ Assamese", "SPAM",
     "অভিনন্দন! আপনি ৫০,০০০ টকা জিকিছে। এতিয়াই দাবী কৰক: 9101234567 ত ফোন কৰক।"),
    ("🏔️ Assamese", "SPAM",
     "বিনামূলীয়া iPhone জিকক! পঞ্জীয়ন কৰক: www.assam-offer.com"),
    # Assamese HAM
    ("🏔️ Assamese", "HAM",
     "কাইলৈ পুৱাতে আহিবা? গাঁৱলৈ যাম।"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { color: #e2e8f0; font-size: 2.2rem; margin: 0; }
.main-header p  { color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0; }

/* ── Result badges ── */
.badge-spam {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white; padding: 1rem 2rem; border-radius: 12px;
    font-size: 1.6rem; font-weight: 700; text-align: center;
    box-shadow: 0 4px 20px rgba(239,68,68,0.4);
}
.badge-ham {
    background: linear-gradient(135deg, #22c55e, #15803d);
    color: white; padding: 1rem 2rem; border-radius: 12px;
    font-size: 1.6rem; font-weight: 700; text-align: center;
    box-shadow: 0 4px 20px rgba(34,197,94,0.4);
}

/* ── Confidence bar ── */
.conf-label { font-size: 0.85rem; color: #64748b; margin-bottom: 4px; }

/* ── Meta pill ── */
.meta-pill {
    display: inline-block; background: #1e293b; color: #94a3b8;
    border-radius: 20px; padding: 4px 14px; font-size: 0.8rem;
    margin: 4px 4px 0 0;
}

/* ── Example buttons ── */
.stButton button {
    width: 100%; text-align: left; font-size: 0.8rem;
    border-radius: 8px; border: 1px solid #334155;
    background: #1e293b; color: #cbd5e1;
    transition: all 0.2s;
}
.stButton button:hover { background: #0f3460; border-color: #3b82f6; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ Multilingual SMS Spam Detector</h1>
  <p>English · Hindi · Bengali · Assamese — powered by MuRIL + LightGBM</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    selected_lang_label = st.selectbox(
        "Language",
        options=list(LANG_OPTIONS.keys()),
        index=0,
        help="Select the SMS language or let the model auto-detect.",
    )
    selected_lang = LANG_OPTIONS[selected_lang_label]

    st.markdown("---")
    st.markdown("## 🏗️ Model Info")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            health = r.json()
            status_icon = "🟢" if health["models_loaded"] else "🔴"
            st.markdown(f"{status_icon} **API Status**: {health['status'].upper()}")
            st.markdown(f"📦 **Version**: `{health['model_version']}`")
            st.markdown(f"🤖 **Models Loaded**: {health['models_loaded']}")
            if health.get("startup_error"):
                st.error(f"Error: {health['startup_error']}")
        else:
            st.warning("API returned unexpected status.")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend not reachable.\nStart it with:\n```\nuvicorn src.api:app --port 8000\n```")

    st.markdown("---")
    st.markdown("## 🏛️ Architecture")
    st.markdown("""
    **Tier 1** — MuRIL Transformer  
    `google/muril-base-cased`  
    Supports EN, HI, BN, AS natively.
    
    **Tier 2** — LightGBM Fallback  
    TF-IDF char n-grams (2–5) + engineered features.  
    Activates when confidence < 0.75 or language unknown.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
col_main, col_examples = st.columns([3, 2], gap="large")

# ── Input panel ────────────────────────────────────────────────────────────
with col_main:
    st.markdown("### ✉️ Enter SMS Text")
    sms_text = st.text_area(
        label="SMS Message",
        value=st.session_state["sms_input"],
        height=160,
        placeholder="Type or paste an SMS message here…",
        label_visibility="collapsed",
        key="sms_text_area",
    )

    predict_col, clear_col = st.columns([3, 1])
    with predict_col:
        predict_btn = st.button("🔍 Classify SMS", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["sms_input"] = ""
            st.session_state["last_result"] = None
            st.rerun()

    # ── Run prediction ──────────────────────────────────────────────────────
    if predict_btn and sms_text.strip():
        with st.spinner("Classifying…"):
            try:
                payload = {"sms": sms_text.strip(), "language": selected_lang}
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
                if resp.status_code == 200:
                    st.session_state["last_result"] = resp.json()
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend. "
                    "Start it with: `uvicorn src.api:app --port 8000`"
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    elif predict_btn and not sms_text.strip():
        st.warning("Please enter an SMS message first.")

    # ── Display result ──────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        st.markdown("### 📊 Classification Result")

        label = result["label"]
        conf  = result["confidence"]

        if label == "SPAM":
            st.markdown(f'<div class="badge-spam">🔴 SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-ham">🟢 HAM (Not Spam)</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="conf-label">Confidence: {conf:.1%}</p>', unsafe_allow_html=True)
        st.progress(conf)

        lang_emoji = {
            "en": "🇬🇧", "hi": "🇮🇳", "bn": "🇧🇩", "as": "🏔️"
        }.get(result["language_detected"], "🌐")
        lang_name = {
            "en": "English", "hi": "Hindi",
            "bn": "Bengali", "as": "Assamese", "unknown": "Unknown"
        }.get(result["language_detected"], result["language_detected"])

        tier_label = "MuRIL (Tier 1)" if result["tier"] == 1 else "LightGBM Fallback (Tier 2)"

        st.markdown(
            f'<span class="meta-pill">{lang_emoji} {lang_name}</span>'
            f'<span class="meta-pill">🤖 {tier_label}</span>'
            f'<span class="meta-pill">⚡ {result["latency_ms"]} ms</span>',
            unsafe_allow_html=True,
        )


# ── Example buttons ─────────────────────────────────────────────────────────
with col_examples:
    st.markdown("### 💬 Quick Examples")
    st.caption("Click any example to load it into the input box.")

    # Group by language
    from itertools import groupby
    for lang_label in ["🇬🇧 English", "🇮🇳 Hindi", "🇧🇩 Bengali", "🏔️ Assamese"]:
        lang_examples = [e for e in EXAMPLES if e[0] == lang_label]
        with st.expander(lang_label, expanded=(lang_label == "🇬🇧 English")):
            for _, badge, text in lang_examples:
                icon = "🔴" if badge == "SPAM" else "🟢"
                short = text[:55] + "…" if len(text) > 55 else text
                if st.button(f"{icon} {short}", key=f"ex_{hash(text)}"):
                    st.session_state["sms_input"] = text
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Multilingual SMS Spam Classifier · MuRIL + LightGBM · EN / HI / BN / AS"
    "</p>",
    unsafe_allow_html=True,
)
