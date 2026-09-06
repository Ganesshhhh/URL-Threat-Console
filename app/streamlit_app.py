"""
Zero-Day Phishing URL Detector -- dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import json
import threading
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import shap

from features import extract_features, FEATURE_NAMES
from adversarial import generate_adversarial_set, evaluate
import ct_feed_worker
from ct_feed_worker import run_simulated, run_live
from phantom_squat import phantom_squat_score, PHANTOM_SQUAT_ILLUSTRATIVE, query_groq_live, POPULAR_BRANDS
from explain import describe_factor  # shared with src/explain.py so the two never drift apart
from model_io import load_model_bundle

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
COMPARISON_PATH = ROOT / "data" / "model_comparison_results.json"
FEED_PATH = ROOT / "data" / "live_feed.jsonl"


@st.cache_resource
def start_live_feed():
    """Starts the real live Certificate Transparency log background listener thread."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_live,
        args=(str(MODEL_PATH), str(FEED_PATH)),
        kwargs={"stop_event": stop_event, "threshold": 0.5},
        daemon=True,
    )
    thread.start()
    return stop_event

# ---------------------------------------------------------------------------
# Styling: cream / graph-paper / terracotta theme, matching the "slumbr"
# reference design -- serif display headings, small-caps eyebrow labels,
# underlined text tabs, boxy cream cards on a faint grid background.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="URL Threat Console", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #eae7d9;
        background-image:
            linear-gradient(rgba(120,120,100,0.09) 1px, transparent 1px),
            linear-gradient(90deg, rgba(120,120,100,0.09) 1px, transparent 1px);
        background-size: 28px 28px;
        color: #3a362c;
    }
    .eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: #b5722f;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .app-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #2c2a22;
        margin: 0;
    }
    .app-tagline {
        color: #7a7566;
        font-size: 0.92rem;
        margin-top: -6px;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: #2c2a22;
    }
    p, li, label, span {
        color: #55503f;
    }
    code, .stCodeBlock, [data-testid="stMetricValue"] {
        font-family: 'Inter', monospace !important;
    }
    [data-testid="stTextInput"] input {
        background-color: #f6f4ea;
        border: 1px solid #d8d3c0;
        border-radius: 6px;
        color: #2c2a22;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 22px;
        border-bottom: 1px solid #d8d3c0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8a8571;
        border-radius: 0;
        padding: 8px 2px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #2c2a22;
        border-bottom: 2px solid #4a4534;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #f3f1e5;
        border: 1px solid #d8d3c0 !important;
        border-radius: 10px;
    }
    .stButton button, .stDownloadButton button {
        background-color: #2c2a22;
        color: #f3f1e5;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton button[kind="primary"] {
        background-color: #b5722f;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #d8d3c0;
        border-radius: 8px;
    }
    .verdict-flagged {
        border: 1px solid #b5544a;
        background-color: #f3e2df;
        color: #8f3b32;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
    .verdict-clear {
        border: 1px solid #5c8462;
        background-color: #e5ecdf;
        color: #3f6446;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
    .factor-row {
        font-size: 0.88rem;
        color: #55503f;
        padding: 4px 0;
        border-bottom: 1px solid #e2ded0;
    }
    hr {
        border-color: #d8d3c0;
    }
    .note-box {
        border-left: 3px solid #b5722f;
        padding: 6px 14px;
        background-color: #f3f1e5;
        color: #7a7566;
        font-size: 0.85rem;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("No trained model found. See README.md for setup steps.")
        st.stop()
    try:
        return load_model_bundle(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model ({e}). Retrain with `compare_models.py`.")
        st.stop()


@st.cache_resource
def load_comparison():
    if not COMPARISON_PATH.exists():
        st.error("No comparison results found. Run `compare_models.py` first.")
        st.stop()
    with open(COMPARISON_PATH) as f:
        return json.load(f)


@st.cache_resource
def start_simulated_feed():
    """Starts the background thread exactly once per server process."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_simulated,
        args=(str(MODEL_PATH), str(FEED_PATH)),
        kwargs={"interval_seconds": 3, "stop_event": stop_event},
        daemon=True,
    )
    thread.start()
    return stop_event


def read_feed(limit=50, source=None):
    if not FEED_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "domain", "phishing_probability", "source"])
    rows = []
    with open(FEED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if source is not None and "source" in df.columns:
        df = df[df["source"] == source]
    if df.empty:
        return df
    return df.sort_values("timestamp", ascending=False).head(limit)


def read_ct_status():
    status_path = FEED_PATH.with_name("ct_status.json")
    if not status_path.exists():
        return {"state": "connecting"}
    try:
        with open(status_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"state": "connecting"}


clf, feature_names = load_model()

st.markdown(
    """
    <div class="eyebrow">URL RISK SCORING & MONITORING</div>
    <div class="app-title">URL Threat Console</div>
    <div class="app-tagline">Catch fakes before they're blocklisted.</div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

tab_score, tab_monitor, tab_phantom, tab_compare = st.tabs(
    ["Score a URL", "Live monitoring", "Phantom squatting", "Model comparison"]
)

# ---------------------------------------------------------------------------
# Tab 1: Score a URL
# ---------------------------------------------------------------------------
with tab_score:
    url = st.text_input("URL to evaluate", placeholder="https://paypa1-secure.tk/verify-account")
    if url:
        feats = extract_features(url)
        X = pd.DataFrame([feats], columns=feature_names)
        prob = clf.predict_proba(X)[0, 1]

        if prob >= 0.5:
            st.markdown(
                f'<div class="verdict-flagged">FLAGGED &nbsp;&nbsp;|&nbsp;&nbsp; phishing probability: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-clear">NOT FLAGGED &nbsp;&nbsp;|&nbsp;&nbsp; phishing probability: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        sv = shap_values[0]
        contributions = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)

        st.subheader("Contributing factors")
        for name, val in contributions.head(5).items():
            direction = "raises" if val > 0 else "lowers"
            readable = describe_factor(name, feats[name])
            st.markdown(
                f'<div class="factor-row">{readable} — {direction} score by {abs(val):.3f}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Full feature vector"):
            st.json(feats)

# ---------------------------------------------------------------------------
# Tab 2: Model comparison
# ---------------------------------------------------------------------------
with tab_compare:
    comparison = load_comparison()
    st.subheader("Why XGBoost")
    st.markdown("Ranked by catch rate on a red-team set, not held-out accuracy alone.")

    summary_df = pd.DataFrame(comparison["summary"]).sort_values(
        "red_team_catch_rate", ascending=False
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="note-box">Chosen model: <b>{comparison["chosen_model"]}</b> — best red-team catch rate.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Catch rate by attack technique")
    per_technique_df = pd.DataFrame(comparison["all_per_technique"])
    st.dataframe(per_technique_df, use_container_width=True)

    st.markdown(
        '<div class="note-box">All models score 0.0 on shorteners — a feature limit, not a model-choice one.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab 3: Live monitoring
# ---------------------------------------------------------------------------
with tab_monitor:
    st.markdown(
        '<div class="note-box">Scores new domains as they appear via Certificate Transparency logs; flags are written to the feed below.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    mode = st.radio(
        "Feed source",
        ["Live Certificate Transparency stream", "Simulated (offline demo / fallback)"],
        horizontal=True,
        index=0,
    )

    # Each panel below is an st.fragment: on its own ~4s timer it reruns only
    # itself, not the whole page. A plain st_autorefresh() reran the ENTIRE
    # script every few seconds -- tearing down and rebuilding the header,
    # tabs, and everything else each time -- which is what made the table
    # flicker in and out and left it "partially visible most of the time".
    # Fragments update just the feed table in place instead.

    @st.fragment(run_every="4s")
    def _live_ct_panel():
        if ct_feed_worker.certstream is None:
            st.error(
                "The `certstream` package is not installed, so the live "
                "Certificate Transparency stream can't start. Run "
                "`pip install certstream` and restart the app, or switch to "
                "the simulated feed above in the meantime."
            )
            return
        start_live_feed()
        status = read_ct_status()
        state = status.get("state", "connecting")
        if state == "connected":
            st.markdown(
                '<div class="note-box"><b>🟢 Live</b> — streaming and scoring new certificates.</div>',
                unsafe_allow_html=True,
            )
        elif state == "socket_open":
            st.markdown(
                '<div class="note-box"><b>🟡 Connected</b> — socket open, waiting for the first certificate.</div>',
                unsafe_allow_html=True,
            )
        elif state == "error":
            st.markdown(
                f'<div class="note-box"><b>🔴 Connection failed</b> — {status.get("error", "unknown error")}. '
                'The public certstream.calidog.io server has a history of outages; try the simulated feed if this persists.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="note-box"><b>🟡 Connecting</b> — opening the certificate stream socket.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        feed_df = read_feed(source="live_ct_log")
        if feed_df.empty:
            st.info("No live-flagged domains yet. Simulated-mode rows are hidden here.")
        else:
            st.dataframe(
                feed_df[["timestamp", "domain", "phishing_probability", "brand_edit_distance", "suspicious_tld", "source"]],
                use_container_width=True,
                hide_index=True,
            )

    @st.fragment(run_every="4s")
    def _simulated_feed_panel():
        start_simulated_feed()
        feed_df = read_feed(source="simulated")
        if feed_df.empty:
            st.write("Waiting for the first flagged domain...")
        else:
            st.dataframe(
                feed_df[["timestamp", "domain", "phishing_probability", "brand_edit_distance", "suspicious_tld", "source"]],
                use_container_width=True,
                hide_index=True,
            )

    if mode == "Live Certificate Transparency stream":
        _live_ct_panel()
    else:
        _simulated_feed_panel()

# ---------------------------------------------------------------------------
# Tab 4: Phantom squatting
# ---------------------------------------------------------------------------
with tab_phantom:
    st.markdown("Catches domains an LLM hallucinates when asked for a brand's URL.")
    
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Small fixed default to keep API usage low -- override with a
    # comma-separated PHANTOM_SQUAT_BRANDS env var if you want more/different
    # brands swept (any name also present in features.POPULAR_BRANDS).
    _default_brands = ["paypal", "microsoft", "chase", "coinbase", "metamask"]
    _env_brands = os.environ.get("PHANTOM_SQUAT_BRANDS")
    if _env_brands:
        sweep_brands = [b.strip().lower() for b in _env_brands.split(",") if b.strip()]
    else:
        sweep_brands = _default_brands

    st.subheader("Live LLM Query via Groq")
    st.markdown(f"Queries `{GROQ_MODEL}` for {len(sweep_brands)} brand(s) and scores the domain it names.")

    @st.cache_data(ttl=600, show_spinner="Querying Groq for all brands...")
    def _run_groq_sweep(api_key: str, model: str, brands: tuple):
        results = []
        for brand in brands:
            try:
                r = query_groq_live(brand=brand, api_key=api_key, model=model)
                ext_url = f"https://{r['extracted_domain']}/"
                prob = clf.predict_proba(
                    pd.DataFrame([extract_features(ext_url)], columns=feature_names)
                )[0, 1]
                results.append({
                    "brand": brand,
                    "raw_response": r["raw_response"],
                    "extracted_domain": r["extracted_domain"],
                    "phantom_squat_likely": r["phantom_squat_analysis"]["phantom_squat_likely"],
                    "heuristic_score": r["phantom_squat_analysis"]["score"],
                    "xgboost_phishing_probability": round(float(prob), 4),
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "brand": brand, "raw_response": None, "extracted_domain": None,
                    "phantom_squat_likely": None, "heuristic_score": None,
                    "xgboost_phishing_probability": None, "error": str(e),
                })
        return results

    if not groq_api_key:
        st.info("`GROQ_API_KEY` not set — live sweep skipped.")
    else:
        results = _run_groq_sweep(groq_api_key, GROQ_MODEL, tuple(sweep_brands))
        results_df = pd.DataFrame(results)
        errors = results_df[results_df["error"].notna()]
        ok = results_df[results_df["error"].isna()].drop(columns=["error"])

        if not ok.empty:
            st.dataframe(ok, use_container_width=True, hide_index=True)
        if not errors.empty:
            with st.expander(f"{len(errors)} brand(s) failed to query"):
                st.dataframe(errors[["brand", "error"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Illustrative Patterns")
    rows = []
    for entry in PHANTOM_SQUAT_ILLUSTRATIVE:
        result = phantom_squat_score(entry["hallucinated_domain"])
        rows.append({
            "domain": entry["hallucinated_domain"],
            "brand": entry["brand"],
            "phantom_squat_likely": result["phantom_squat_likely"],
            "score": result["score"],
            "reason": result["reason"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Check a domain manually")
    domain_input = st.text_input("Domain to check", placeholder="paypalsupport.com")
    if domain_input:
        result = phantom_squat_score(domain_input)
        st.json(result)
