# app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
from pathlib import Path
from src.predict import predict, validate_inputs
from src.weather import fetch_location_weather, weather_description

st.set_page_config(
    page_title="AgriSense — Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "crop_data.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,300;1,9..144,400&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
/* Streamlit injects gap/margin between markdown blocks via flex containers —
   the universal reset above has 0 specificity and loses to Streamlit's own
   class-based rules, which is what causes the dark strip between hero and content */
div.element-container:has(.hero-wrap) { margin: 0 !important; }
div[data-testid="stVerticalBlock"]:has(> div.element-container:has(.hero-wrap)) { gap: 0 !important; }
div[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
div[data-testid="stToolbar"] { display: none !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── HERO ── */
.hero-wrap {
    position: relative;
    width: 100%;
    min-height: 92vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #0d1f14;
}
.hero-img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; opacity: 0.45;
    filter: saturate(1.1) contrast(1.05);
}
.hero-overlay {
    position: absolute; inset: 0;
    background:
        linear-gradient(to bottom, rgba(5,15,8,0.3) 0%, rgba(5,15,8,0.1) 40%, rgba(5,15,8,0.65) 100%),
        linear-gradient(to right, rgba(5,15,8,0.5) 0%, transparent 60%);
}
.hero-content {
    position: relative; z-index: 2;
    max-width: 760px; padding: 4rem 3rem;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(134,220,120,0.18);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(134,220,120,0.4);
    color: #a8f0a0; font-size: 11px; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase;
    padding: 6px 16px; border-radius: 100px; margin-bottom: 1.5rem;
}
.hero-tag::before { content: ''; width:7px; height:7px; background:#6ee86e; border-radius:50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }

.hero-title {
    font-family: 'Fraunces', serif; font-size: clamp(42px, 6vw, 72px);
    font-weight: 700; color: #f4f9f2; line-height: 1.08;
    letter-spacing: -1.5px; margin-bottom: 1.25rem;
}
.hero-title em { color: #86dc78; font-style: italic; font-weight: 300; }
.hero-sub {
    font-size: 17px; color: rgba(255,255,255,0.7); font-weight: 300;
    line-height: 1.75; max-width: 520px; margin-bottom: 2.5rem;
}
.hero-actions { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
.btn-primary {
    background: #86dc78; color: #0d2010; font-weight: 600; font-size: 15px;
    padding: 14px 32px; border-radius: 100px; border: none; cursor: pointer;
    text-decoration: none; display: inline-block;
    transition: all 0.2s; letter-spacing: -0.2px;
}
.btn-primary:hover { background: #a4ec96; transform: translateY(-1px); }
.btn-ghost {
    background: transparent; color: rgba(255,255,255,0.8); font-weight: 500;
    font-size: 15px; padding: 14px 28px; border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.3); cursor: pointer;
    text-decoration: none; display: inline-block; transition: all 0.2s;
}
.btn-ghost:hover { border-color: rgba(255,255,255,0.7); color: #fff; }

.hero-stats {
    position: absolute; bottom: 2.5rem; left: 3rem; right: 3rem;
    z-index: 2; display: flex; gap: 2rem; flex-wrap: wrap;
}
.hero-stat { border-left: 2px solid rgba(134,220,120,0.5); padding-left: 1rem; }
.hero-stat-val { font-family:'Fraunces',serif; font-size:28px; font-weight:700; color:#f4f9f2; line-height:1; }
.hero-stat-lbl { font-size:11px; color:rgba(255,255,255,0.5); letter-spacing:1px; text-transform:uppercase; margin-top:4px; }

/* ── MAIN CONTENT ── */
.main-wrap { max-width: 1140px; margin: 0 auto; padding: 5rem 2rem; }

/* Section heading */
.sec-eyebrow {
    font-size:11px;
    font-weight:600;
    letter-spacing:3px;
    text-transform:uppercase;
    color:#86dc78;
    margin-bottom:.5rem;
}

.sec-title {
    font-family:'Fraunces',serif;
    font-size:clamp(28px,3.5vw,42px);
    font-weight:700;
    color:#ffffff;
    line-height:1.15;
    letter-spacing:-0.5px;
    margin-bottom:1rem;
}

.sec-sub {
    font-size:15px;
    color:rgba(255,255,255,0.75);
    line-height:1.7;
    max-width:560px;
}
/* ── MODE TOGGLE ── */
.mode-tabs { display:flex; gap:0; border-radius:14px; overflow:hidden; border:1.5px solid #c8e6c4; margin-bottom:2.5rem; }
.mode-tab { flex:1; padding:1rem 1.5rem; background:#fff; cursor:pointer; transition:all .2s; text-align:center; }
.mode-tab:first-child { border-right:1.5px solid #c8e6c4; }
.mode-tab.active { background:#0d2010; }
.mode-tab-icon { font-size:22px; display:block; margin-bottom:.3rem; }
.mode-tab-title { font-size:14px; font-weight:600; color:#0d2010; }
.mode-tab.active .mode-tab-title { color:#86dc78; }
.mode-tab-sub { font-size:11px; color:#7a9a80; margin-top:.2rem; }
.mode-tab.active .mode-tab-sub { color:rgba(255,255,255,.5); }

/* ── FORM CARDS ── */
.form-section { background:#1a2e1d; border:1.5px solid #2d4a30; border-radius:20px; padding:1.25rem; margin-bottom:1rem; }
.form-section-title { font-size:11px; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:#82dc6e; margin-bottom:1.5rem; padding-bottom:.75rem; border-bottom:1.5px solid #2d4a30; }

/* Farmer question cards */
.q-card { background:#16261a; border:1px solid #2d4a30; border-radius:14px; padding:1.25rem 1.5rem; margin-bottom:1rem; }
.q-label { font-size:13px; font-weight:600; color:#ffffff; margin-bottom:.5rem; }
.q-hint  { font-size:11px; color:rgba(255,255,255,0.65); margin-bottom:.75rem; line-height:1.5; }

/* Info callout */
.callout { display:flex; gap:1rem; align-items:flex-start; background:#1a2e1d; border:1px solid #2d4a30; border-radius:14px; padding:1.25rem 1.5rem; margin-bottom:2rem; }
.callout-icon { font-size:20px; flex-shrink:0; }
.callout-text { font-size:13px; color:#c8e6c9; line-height:1.6; }
.callout-text strong { color:#82dc6e; }

/* Estimated values strip */
.est-strip { background:#0d2010; border-radius:16px; padding:1.5rem 2rem; margin:1.5rem 0; display:grid; grid-template-columns:repeat(7,1fr); gap:1rem; }
.est-item { text-align:center; }
.est-val { font-family:'Fraunces',serif; font-size:22px; font-weight:700; color:#86dc78; display:block; }
.est-lbl { font-size:10px; color:rgba(255,255,255,.5); letter-spacing:1px; text-transform:uppercase; margin-top:3px; }

/* ── RESULT ── */
.result-hero {
    background: linear-gradient(135deg,#0d2010 0%,#1a4a2e 50%,#0d3018 100%);
    border-radius: 24px; padding: 3rem; margin-bottom: 2rem;
    display: flex; align-items: center; gap: 3rem;
    position: relative; overflow: hidden;
}
.result-hero::before {
    content:''; position:absolute; right:-60px; top:-60px;
    width:300px; height:300px; border-radius:50%;
    background:radial-gradient(circle,rgba(134,220,120,.15) 0%,transparent 70%);
}

.result-crop {
    font-family:'Fraunces',serif;
    font-size:58px;
    font-weight:700;
    color:#f4f9f2;
    line-height:1;
    letter-spacing:-2px;
}
.result-conf-badge { display:inline-block; background:rgba(134,220,120,.2); border:1px solid rgba(134,220,120,.4); color:#86dc78; font-size:13px; font-weight:600; padding:5px 16px; border-radius:100px; margin-top:.75rem; }
.result-label { font-size:11px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,.4); margin-bottom:.5rem; }

.conf-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2rem; }
.conf-card { background:#1a2e1d; border:1.5px solid #2d4a30; border-radius:16px; padding:1.25rem; text-align:center; transition:transform .25s ease, box-shadow .25s ease, border-color .2s; }
.conf-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.35); border-color:#86dc78; }
.conf-card:first-child { border-color:#86dc78; }
.conf-rank { font-size:10px; color:#82dc6e; letter-spacing:2px; text-transform:uppercase; margin-bottom:.35rem; }
.conf-crop { font-family:'Fraunces',serif; font-size:18px; font-weight:700; color:#f0f8ee; text-transform:capitalize; margin-bottom:.25rem; }
.conf-pct  { font-size:26px; font-weight:700; color:#82dc6e; }
.conf-bar-bg { background:#e8f4ea; border-radius:100px; height:4px; margin-top:.75rem; overflow:hidden; }
.conf-bar    { background:linear-gradient(90deg,#2d7a4f,#86dc78); height:100%; border-radius:100px; }

/* ── SUMMARY TABLE ── */
.summary-table { width:100%; border-collapse:collapse; }
.summary-table th { background:#0d2010; color:rgba(255,255,255,.7); font-size:10px; letter-spacing:1.5px; text-transform:uppercase; padding:.6rem 1rem; text-align:left; }
.summary-table td { padding:.6rem 1rem; font-size:13px; color:#e0f0e0; border-bottom:1px solid #2d4a30; }
.summary-table tr:last-child td { border-bottom:none; }
.summary-table tr:hover td { background:#1f3522; }

/* ── WIKI ── */
.wiki-wrap { background:#1a2e1d; border:1.5px solid #2d4a30; border-radius:18px; padding:1.5rem; height:100%; }
.wiki-name { font-family:'Fraunces',serif; font-size:20px; color:#f0f8ee; margin-bottom:.6rem; }
.wiki-desc { font-size:13px; color:#c8e6c9; line-height:1.75; margin-bottom:1rem; }
.wiki-link { font-size:12px; font-weight:600; color:#2d7a4f; text-decoration:none; }

/* ── PERFORMANCE SECTION ── */
.perf-wrap { background:#0d2010; border-radius:24px; padding:3rem; margin-bottom:2rem; color:white; }
.acc-number { font-family:'Fraunces',serif; font-size:80px; font-weight:700; color:#86dc78; line-height:1; }
.acc-suffix { font-size:32px; color:rgba(134,220,120,.6); }
.acc-label  { font-size:11px; letter-spacing:3px; color:rgba(255,255,255,.4); text-transform:uppercase; }
.acc-note   { font-size:13px; color:rgba(255,255,255,.55); line-height:1.6; max-width:400px; margin-top:.75rem; }

/* ── CHART WRAPPER ── */
.chart-card { background:#1a2e1d; border:1.5px solid #2d4a30; border-radius:20px; padding:2rem; margin-bottom:1.5rem; transition:box-shadow .25s ease; }
.chart-card:hover { box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
.chart-title { font-family:'Fraunces',serif; font-size:20px; color:#f0f8ee; margin-bottom:.3rem; }
.chart-sub   { font-size:13px; color:#82dc6e; margin-bottom:1.25rem; }

/* ── CTA strip at bottom ── */
.cta-strip { background:linear-gradient(135deg,#0d2010,#1a4a2e); border-radius:24px; padding:3rem; text-align:center; margin-top:4rem; }
.cta-title { font-family:'Fraunces',serif; font-size:32px; color:#f4f9f2; margin-bottom:.75rem; }
.cta-sub { font-size:15px; color:rgba(255,255,255,.6); margin-bottom:1.75rem; }

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label { display:none !important; }
.stButton > button { border-radius:100px !important; font-weight:600 !important; }
.stButton > button[kind="primary"] {
    background:#86dc78 !important; color:#0d2010 !important;
    border:none !important; font-size:15px !important; padding:.75rem 2rem !important;
}
.stButton > button[kind="secondary"] {
    background:transparent !important; border:1.5px solid #c8e6c4 !important;
    color:#2d7a4f !important;
}
div[data-baseweb="select"] > div { border-radius:10px !important; border-color:#3d6b42 !important; background:#1f3522 !important; color:#e0f0e0 !important; }
div[data-baseweb="input"] > div   { border-radius:10px !important; border-color:#3d6b42 !important; background:#1f3522 !important; color:#e0f0e0 !important; }

@media (max-width: 768px) {
    .hero-content { padding: 2.5rem 1.5rem; }
    .hero-stats { position: static; padding: 1.5rem; flex-direction: column; gap: 1rem; }
    .hero-wrap { min-height: auto; padding-bottom: 1rem; }

    .est-strip { grid-template-columns: repeat(2, 1fr); }
    .conf-grid { grid-template-columns: repeat(2, 1fr); }

    .result-hero { flex-direction: column; text-align: center; gap: 1.5rem; padding: 2rem 1.5rem; }
    .result-crop { font-size: 38px; }

    .main-wrap { padding: 2.5rem 1.25rem; }

    .summary-table { display: block; overflow-x: auto; white-space: nowrap; }
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        'model':    joblib.load(MODEL_DIR / 'rf_crop_model.joblib'),
        'scaler':   joblib.load(MODEL_DIR / 'scaler.joblib'),
        'le':       joblib.load(MODEL_DIR / 'label_encoder.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib'),
        'cm':       joblib.load(MODEL_DIR / 'confusion_matrix.joblib')
                    if (MODEL_DIR/'confusion_matrix.joblib').exists() else None,
        'test_acc': joblib.load(MODEL_DIR / 'test_accuracy.joblib')
                    if (MODEL_DIR/'test_accuracy.joblib').exists() else None,
    }

artifacts     = load_models()
model         = artifacts['model']
scaler        = artifacts['scaler']
label_encoder = artifacts['le']
feature_names = artifacts['features']
@st.cache_data
def _get_dataset_rows():
    try: return pd.read_csv(DATA_PATH).shape[0]
    except: return 2200

dataset_rows = _get_dataset_rows()

# ── Farmer mode mappings ──────────────────────────────────────────────────────
SOIL_MAP = {
    "Black (Dark / Cotton soil)":  {"N":80,"P":60,"K":80,"ph":7.2},
    "Red / Laterite":              {"N":40,"P":30,"K":50,"ph":6.0},
    "Sandy / Light brown":         {"N":25,"P":20,"K":30,"ph":6.5},
    "Clay / Heavy / Sticky":       {"N":70,"P":55,"K":75,"ph":7.5},
    "Loamy / Mixed (best soil)":   {"N":60,"P":45,"K":60,"ph":6.8},
    "Alluvial (near river)":       {"N":90,"P":65,"K":85,"ph":7.0},
}
LAST_CROP_MAP = {
    "Rice":                 {"N_adj":-15,"P_adj":-10,"K_adj":-20},
    "Wheat":                {"N_adj":-10,"P_adj":-8, "K_adj":-12},
    "Maize / Corn":         {"N_adj":-12,"P_adj":-9, "K_adj":-15},
    "Cotton":               {"N_adj":-20,"P_adj":-15,"K_adj":-25},
    "Sugarcane":            {"N_adj":-25,"P_adj":-12,"K_adj":-30},
    "Vegetables":           {"N_adj":-8, "P_adj":-6, "K_adj":-10},
    "Pulses / Lentils":     {"N_adj":10, "P_adj":5,  "K_adj":5},
    "Nothing (fallow land)":{"N_adj":5,  "P_adj":3,  "K_adj":5},
}
RAIN_MAP = {
    "Very dry  (< 500 mm/yr)":    {"rainfall":50, "humidity":28},
    "Semi-arid (500–1000 mm/yr)": {"rainfall":120,"humidity":48},
    "Moderate  (1000–1500 mm/yr)":{"rainfall":200,"humidity":68},
    "Humid     (1500–2500 mm/yr)":{"rainfall":250,"humidity":84},
    "Very wet  (> 2500 mm/yr)":   {"rainfall":290,"humidity":95},
}
TEMP_MAP = {
    "Cool — Hills / Winter (< 15 °C)": 13.0,
    "Mild — Spring / Autumn (15–25 °C)":22.0,
    "Warm — Plains / Summer (25–35 °C)":30.0,
    "Hot  — Arid / Peak summer (> 35 °C)":40.0,
}

CROP_EMOJI = {"rice":"🌾","wheat":"🌾","maize":"🌽","apple":"🍎","banana":"🍌",
              "mango":"🥭","papaya":"🍈","cotton":"🌸","jute":"🌿","grapes":"🍇",
              "orange":"🍊","coconut":"🥥","coffee":"☕","chickpea":"🫘",
              "kidneybeans":"🫘","lentil":"🫘","blackgram":"🫘","mungbean":"🫘",
              "mothbeans":"🫘","pigeonpeas":"🫘","muskmelon":"🍈","watermelon":"🍉",
              "pomegranate":"🍎"}

# Map dataset crop keys -> real, correctly-titled Wikipedia article names
WIKI_TITLE_MAP = {
    "kidneybeans":"Kidney bean", "mungbean":"Mung bean", "blackgram":"Black gram",
    "mothbeans":"Moth bean", "pigeonpeas":"Pigeon pea", "muskmelon":"Muskmelon",
    "chickpea":"Chickpea", "lentil":"Lentil", "watermelon":"Watermelon",
    "pomegranate":"Pomegranate", "papaya":"Papaya", "coconut":"Coconut",
}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wiki(name):
    title = WIKI_TITLE_MAP.get(name.lower(), name).replace(' ', '_')
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                         timeout=5, headers={"User-Agent":"agrisense/1.0"})
        if r.status_code != 200: return None
        d = r.json()
        return {"description":d.get("extract",""), "image":d.get("thumbnail",{}).get("source")}
    except: return None

# Fallback text + fallback image for every crop the model can predict
FALLBACK = {
    "rice":"Staple cereal grown in flooded paddies. Requires warm, wet conditions.",
    "wheat":"Temperate cereal; the base of flour and bread worldwide.",
    "maize":"Versatile corn crop used for food, feed, and biofuel.",
    "chickpea":"A protein-rich legume grown widely in semi-arid regions.",
    "kidneybeans":"A nutrient-dense legume valued for its protein content.",
    "pigeonpeas":"A drought-tolerant legume used widely in South Asian cuisine.",
    "mothbeans":"A heat- and drought-resistant legume suited to arid soils.",
    "mungbean":"A fast-growing legume used for food and soil nitrogen fixing.",
    "blackgram":"A legume crop common in South Asian cooking, rich in protein.",
    "lentil":"A small, protein-rich legume grown in cooler, drier climates.",
    "pomegranate":"A fruit-bearing shrub thriving in semi-arid, hot climates.",
    "banana":"A tropical fruit crop needing consistent warmth and humidity.",
    "mango":"A tropical fruit tree requiring a warm, frost-free climate.",
    "grapes":"A fruiting vine grown in temperate to subtropical climates.",
    "watermelon":"A warm-season fruit crop needing long, hot growing seasons.",
    "muskmelon":"A warm-climate fruit related to cantaloupe and honeydew.",
    "apple":"A temperate fruit tree requiring a cold winter dormancy period.",
    "orange":"A subtropical citrus fruit needing mild winters and warm summers.",
    "papaya":"A fast-growing tropical fruit tree sensitive to frost.",
    "coconut":"A tropical palm thriving in coastal, humid environments.",
    "cotton":"A fiber crop requiring a long, warm growing season.",
    "jute":"A fiber crop grown in hot, humid, monsoon-fed regions.",
    "coffee":"A tropical shrub grown at altitude in warm, humid climates.",
}

# Static fallback images so a crop card is never empty if Wikipedia is unreachable
FALLBACK_IMG = {
    "rice":"https://images.unsplash.com/photo-1568347355280-d33fdb1ecb56?w=600&q=80",
    "coffee":"https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=600&q=80",
    "cotton":"https://images.unsplash.com/photo-1594824476967-48c8b964273f?w=600&q=80",
    # add more as desired — this dict is just consulted when wiki image is None
}


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <img class="hero-img"
         src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1600&q=80&fit=crop"
         alt="Golden wheat field at sunrise" />
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <div class="hero-tag">AI-Powered Agriculture</div>
        <h1 class="hero-title">
            Grow the right crop,<br>
            <em>every single season.</em>
        </h1>
        <p class="hero-sub">
            AgriSense analyses your soil nutrients and climate conditions
            to recommend the most profitable crop — backed by machine learning
            trained on 2,200 real field samples.
        </p>
        <div class="hero-actions">
            <a href="#crop-input" class="btn-primary">
                Get Started Below ↓
            </a>
            <a href="#model-stats" class="btn-ghost">
                View Model Stats
            </a>
        </div>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">99.55%</div>
            <div class="hero-stat-lbl">Test Accuracy</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">22</div>
            <div class="hero-stat-lbl">Crop Varieties</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">2,200</div>
            <div class="hero-stat-lbl">Training Samples</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">2</div>
            <div class="hero-stat-lbl">Input Modes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# Anchor for Get Started button
st.markdown(
    '<div id="crop-input"></div>',
    unsafe_allow_html=True
)

# Section heading
st.markdown("""
<div style="text-align:center; margin-bottom:3rem;">
    <p class="sec-eyebrow" style="text-align:center;">Step 1</p>
    <h2 class="sec-title" style="text-align:center; margin:0 auto .75rem;">How would you like to enter your field data?</h2>
    <p class="sec-sub" style="text-align:center; margin:0 auto;">
        Choose Farmer Mode if you don't have a soil test report.
        Choose Expert Mode if you have exact soil lab values.
    </p>
</div>
""", unsafe_allow_html=True)

# Mode toggle
mode = st.radio("mode", ["👨‍🌾  Farmer Mode", "🔬  Expert Mode"], horizontal=True, label_visibility="collapsed")
farmer_mode = "Farmer" in mode

if farmer_mode:
    st.markdown("""
    <div class="callout">
        <span class="callout-icon">💡</span>
        <p class="callout-text">
            <strong>No soil testing kit required.</strong>
            Just look at your field and answer 5 simple questions — we'll calculate
            the soil chemistry automatically using agricultural research data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="form-section"><p class="form-section-title">🌍 Your Soil</p>', unsafe_allow_html=True)
        st.markdown('<p class="q-label">What does your soil look like?</p><p class="q-hint">Pick the colour and texture closest to your field.</p>', unsafe_allow_html=True)
        soil_color = st.selectbox("soil", list(SOIL_MAP.keys()), label_visibility="collapsed")
        st.markdown('<p class="q-label" style="margin-top:.75rem;">What did you grow last season?</p><p class="q-hint">Previous crops leave nutrients behind — or take them away.</p>', unsafe_allow_html=True)
        last_crop = st.selectbox("crop", list(LAST_CROP_MAP.keys()), label_visibility="collapsed")
        st.markdown('<p class="q-label" style="margin-top:.75rem;">Do you use fertilizer or manure?</p>', unsafe_allow_html=True)
        fertilizer = st.selectbox("fert", ["Yes — regularly","Sometimes","Rarely or never"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="form-section"><p class="form-section-title">🌤️ Live Weather + Climate</p>',
            unsafe_allow_html=True
        )

        location = st.text_input(
            "📍 Enter your city or village",
            placeholder="e.g. Makthal, Warangal, Hyderabad"
        )

        st.markdown(
            '<p class="q-label">How much rain does your area get?</p><p class="q-hint">Think about a full year of rainfall, not just one season.</p>',
            unsafe_allow_html=True
        )

        rain_choice = st.selectbox(
            "rain",
            list(RAIN_MAP.keys()),
            index=2,
            label_visibility="collapsed"
        )

        st.markdown(
            '<p class="q-label" style="margin-top:.75rem;">What temperature is it when you plant?</p><p class="q-hint">Pick the season or conditions at planting time.</p>',
            unsafe_allow_html=True
        )

        temp_choice = st.selectbox(
            "temp",
            list(TEMP_MAP.keys()),
            index=2,
            label_visibility="collapsed"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Calculate
    base = SOIL_MAP[soil_color].copy()
    adj  = LAST_CROP_MAP[last_crop]
    rain = RAIN_MAP[rain_choice]
    temp = TEMP_MAP[temp_choice]
    if location.strip():
        with st.spinner(f"📡 Finding {location}..."):
            weather = fetch_location_weather(location.strip())
        if weather is None and location.strip():
            st.warning(
                "📍 Location lookup temporarily unavailable. "
                "Using your dropdown selections instead."
            )
    else:
        weather = None
     
    fb   = {"Yes — regularly":1.2,"Sometimes":1.0,"Rarely or never":0.8}[fertilizer]

    N_e  = max(5,  min(140, int((base["N"]+adj["N_adj"])*fb)))
    P_e  = max(5,  min(145, int((base["P"]+adj["P_adj"])*fb)))
    K_e  = max(5,  min(205, int((base["K"]+adj["K_adj"])*fb)))
    ph_e = base["ph"]
    rain_e = rain["rainfall"]

    if weather:
        temp_e = float(weather.get("temperature", temp))
        hum_e  = float(weather.get("humidity", rain["humidity"]))
        if weather.get("rainfall") is not None:
            rain_e = float(weather["rainfall"])
    else:
        temp_e = temp
        hum_e = rain["humidity"]
        
    if weather:
        desc, emoji = weather_description(
            weather.get("weather_code", 0)
        )
        st.success(
            f"📍 {weather['display_name'].split(',')[0]} | "
            f"{emoji} {desc} | "
            f"🌡️ {temp_e:.1f}°C | "
            f"💧 {hum_e:.1f}%"
        )
        

    st.markdown(f"""
    <div class="est-strip">
        <div class="est-item"><span class="est-val">{N_e}</span><span class="est-lbl">Nitrogen</span></div>
        <div class="est-item"><span class="est-val">{P_e}</span><span class="est-lbl">Phosphorus</span></div>
        <div class="est-item"><span class="est-val">{K_e}</span><span class="est-lbl">Potassium</span></div>
        <div class="est-item"><span class="est-val">{temp_e:.1f}°</span><span class="est-lbl">Temp (°C)</span></div>
        <div class="est-item"><span class="est-val">{hum_e:.1f}%</span><span class="est-lbl">Humidity</span></div>
        <div class="est-item"><span class="est-val">{ph_e}</span><span class="est-lbl">Soil pH</span></div>
        <div class="est-item"><span class="est-val">{rain_e}</span><span class="est-lbl">Rain (mm)</span></div>
    </div>
    <p style="font-size:12px;color:#7a9a80;text-align:center;margin-bottom:1.5rem;">
        ↑ Estimated values calculated from your answers — used as model inputs
    </p>
    """, unsafe_allow_html=True)

    values_to_predict = [N_e, P_e, K_e, temp_e, hum_e, ph_e, rain_e]

else:
    st.markdown("""
    <div class="callout">
        <span class="callout-icon">🔬</span>
        <p class="callout-text">
            <strong>Expert Mode.</strong>
            Enter values from your soil test report or IoT sensors.
            All ranges are based on the model's training data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="form-section"><p class="form-section-title">🌱 Soil Nutrients</p>', unsafe_allow_html=True)
        N  = st.number_input("Nitrogen — N (kg/ha)   [0–140]",   0,   140,  90,  1)
        P  = st.number_input("Phosphorus — P (kg/ha) [5–145]",   5,   145,  42,  1)
        K  = st.number_input("Potassium — K (kg/ha)  [5–205]",   5,   205,  43,  1)
        ph = st.number_input("Soil pH                [3.5–9.95]", 3.5, 9.95, 6.5, 0.01, format="%.2f")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="form-section"><p class="form-section-title">🌦️ Climate Data</p>', unsafe_allow_html=True)
        temperature = st.number_input("Temperature (°C)    [8–44]",   8.0, 44.0,  25.0, 0.1, format="%.1f")
        humidity    = st.number_input("Humidity (%)        [14–100]", 14.0,100.0,  80.0, 0.1, format="%.1f")
        rainfall    = st.number_input("Annual Rainfall (mm)[20–299]", 20.0,299.0, 200.0, 1.0, format="%.0f")
        st.markdown('</div>', unsafe_allow_html=True)

    values_to_predict = [N, P, K, temperature, humidity, ph, rainfall]

# Predict button
st.markdown('<div style="text-align:center;margin:1rem 0 2rem;">', unsafe_allow_html=True)
predict_btn = st.button("🌱  Analyse My Field & Recommend a Crop", type="primary", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# ── Result ─────────────────────────────────────────────────────────────────────
if predict_btn:
    errors = validate_inputs(values_to_predict)
    if errors:
        st.error("Some values are outside the valid range:\n" + "\n".join(f"- {e}" for e in errors))
    else:
        with st.spinner("Running model..."):
            result = predict(values_to_predict, top_n=4)

        if 'error' in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            pairs = result['probabilities']
            top_crop, top_conf = pairs[0]
            if top_conf < 50:
                st.warning(
                    f"⚠️ Low confidence prediction ({top_conf:.1f}%). "
                    "Consider verifying with a soil test for better accuracy."
                )
            elif top_conf < 75:
                st.info(
                   f"ℹ️ Moderate confidence ({top_conf:.1f}%). "
                   "Result is likely correct but a soil test can confirm."
                )
            emoji = CROP_EMOJI.get(top_crop.lower(), "🌿")

            st.markdown(f"""
            <div class="result-hero">
                <div style="font-size:72px;flex-shrink:0">{emoji}</div>
                <div>
                    <p class="result-label">Best crop for your field</p>
                    <p class="result-crop">{top_crop.title()}</p>
                    <span class="result-conf-badge">✓ {top_conf:.1f}% confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence grid
            ranks = ["1st","2nd","3rd","4th"]
            html = '<div class="conf-grid">'
            for rank,(crop,pct) in zip(ranks,pairs):
                e = CROP_EMOJI.get(crop.lower(),"🌿")
                html += f"""<div class="conf-card">
                    <p class="conf-rank">{rank} choice</p>
                    <p style="font-size:28px;margin:.2rem 0">{e}</p>
                    <p class="conf-crop">{crop.title()}</p>
                    <p class="conf-pct">{pct:.1f}%</p>
                    <div class="conf-bar-bg"><div class="conf-bar" style="width:{int(min(pct,100))}%"></div></div>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            # Table + wiki
            tc, wc = st.columns([1.1, 1], gap="large")
            labels = ["Nitrogen (N)","Phosphorus (P)","Potassium (K)","Temperature","Humidity","Soil pH","Rainfall"]
            units  = ["kg/ha","kg/ha","kg/ha","°C","%","","mm"]
            with tc:
                st.markdown('<div class="chart-card"><p class="chart-title">Field Input Summary</p>', unsafe_allow_html=True)
                tbl = '<table class="summary-table"><tr><th>Parameter</th><th>Unit</th><th>Value</th></tr>'
                for lbl,unit,val in zip(labels,units,values_to_predict):
                    tbl += f'<tr><td>{lbl}</td><td style="color:#7a9a80">{unit}</td><td style="font-weight:600">{val}</td></tr>'
                tbl += '</table>'
                st.markdown(tbl + '</div>', unsafe_allow_html=True)

            with wc:
                wiki = fetch_wiki(top_crop)
                desc = (wiki.get("description") if wiki else None) or FALLBACK.get(top_crop.lower(),"")
                img_url = (wiki.get("image") if wiki else None) or FALLBACK_IMG.get(top_crop.lower())
                if img_url:
                    try: st.image(img_url, use_container_width=True)
                    except: pass
                st.markdown(f"""
                <div class="wiki-wrap">
                    <p class="wiki-name">{emoji} {top_crop.title()}</p>
                    <p class="wiki-desc">{desc[:320]}{'…' if len(desc)>320 else ''}</p>
                    <a class="wiki-link" href="https://en.wikipedia.org/wiki/{top_crop.replace(' ','_')}" target="_blank">
                        Read full article on Wikipedia →
                    </a>
                </div>""", unsafe_allow_html=True)
                st.markdown("### 🧠 Why This Crop?")
                
                st.success(f"✅ Temperature: {values_to_predict[3]}°C")
                st.success(f"✅ Humidity: {values_to_predict[4]}%")
                st.success(f"✅ Soil pH: {values_to_predict[5]}")
                st.success(f"✅ Rainfall: {values_to_predict[6]} mm")
                st.info(
                    f"The model recommended **{top_crop.title()}** because the soil nutrients "
                    f"and climate conditions closely match patterns learned during training."
              )



st.markdown(
    '<div id="model-stats"></div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:5rem;margin-bottom:2rem;">
    <p class="sec-eyebrow">Performance</p>
    <h2 class="sec-title">Model Analytics</h2>
    <p class="sec-sub">Evaluated on data the model has never seen during training.</p>
</div>
""", unsafe_allow_html=True)

if artifacts['test_acc'] is not None:
    acc = artifacts['test_acc']
    ac, tc2 = st.columns([1, 1.6], gap="large")
    with ac:
        st.markdown(f"""
        <div class="perf-wrap">
            <p class="acc-label">Test Set Accuracy</p>
            <p class="acc-number">{acc*100:.1f}<span class="acc-suffix">%</span></p>
            <p class="acc-note">
                Measured on {int(dataset_rows*0.2):,} held-out samples.
                The model correctly identified the right crop
                <strong style="color:#86dc78">{int(acc*int(dataset_rows*0.2)):,} times
                out of {int(dataset_rows*0.2):,}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown('<div class="chart-card"><p class="chart-title">Feature Importance</p><p class="chart-sub">Which soil/climate factors matter most?</p>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({'Feature':list(feature_names),'Importance':model.feature_importances_}).sort_values('Importance',ascending=True)
        fig,ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        mx = fi_df['Importance'].max()
        cols_fi = ['#0d2010' if v==mx else '#2d7a4f' if v>fi_df['Importance'].median() else '#a8dbb8' for v in fi_df['Importance']]
        bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=cols_fi, height=0.55)
        for bar,val in zip(bars,fi_df['Importance']):
            ax.text(bar.get_width()+.003, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va='center', fontsize=9, color='#1a3a28')
        ax.set_xlim(0, mx*1.22); ax.set_xlabel("Importance", fontsize=10, color='#6a8a75')
        ax.tick_params(colors='#1a3a28', labelsize=10)
        for s in ax.spines.values(): s.set_visible(False)
        ax.xaxis.grid(True, color='#eef5ef', linewidth=0.8); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

cm_expander = st.expander("📊 Advanced Model Metrics")
if cm_expander.expanded and artifacts['cm'] is not None:
    with cm_expander:
        st.markdown(
            '<div class="chart-card"><p class="chart-title">Confusion Matrix — Normalized (Test Set)</p><p class="chart-sub">Diagonal = correct predictions. 1.00 = perfect recall for that crop.</p>',
            unsafe_allow_html=True
        )

        cm_norm = artifacts['cm'].astype(float) / artifacts['cm'].sum(axis=1, keepdims=True)

        fig2, ax2 = plt.subplots(figsize=(14,10))
        fig2.patch.set_facecolor('white')

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="YlGn",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            linewidths=0.4,
            linecolor='#f0f8f2',
            vmin=0,
            vmax=1,
            ax=ax2,
            annot_kws={"size":8}
        )

        ax2.set_xlabel(
            "Predicted Crop",
            fontsize=12,
            fontweight='600',
            color='#0d2010',
            labelpad=10
        )

        ax2.set_ylabel(
            "Actual Crop",
            fontsize=12,
            fontweight='600',
            color='#0d2010',
            labelpad=10
        )

        ax2.set_title("", pad=0)

        plt.xticks(
            rotation=45,
            ha='right',
            fontsize=9,
            color='#1a3a28'
        )

        plt.yticks(
            fontsize=9,
            color='#1a3a28'
        )

        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown('</div>', unsafe_allow_html=True)

# CTA
st.markdown("""
<div class="cta-strip">
    <h2 class="cta-title">Ready to try it on your field?</h2>
    <p class="cta-sub">Scroll up, choose your input mode, and get your personalised crop recommendation in seconds.</p>
    <a class="btn-primary" href="#" style="display:inline-block;">Get Recommendation ↑</a>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrap
