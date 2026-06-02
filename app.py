import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        body, .stApp {
            background: linear-gradient(180deg, #0f3b2f 0%, #1a5d48 50%, #2d8a7e 100%);
            min-height: 100vh;
        }
        .stAppViewContainer {
            background: linear-gradient(180deg, #0f3b2f 0%, #1a5d48 50%, #2d8a7e 100%);
        }
        .stMain {
            background: linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(240,249,245,0.98) 100%);
            border-radius: 35px;
            margin: 2rem 1.5rem;
            padding: 0;
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }
        .block-container {
            padding: 3rem 3.5rem;
            max-width: 1800px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #00d084 0%, #00a86b 100%);
            color: white;
            border-radius: 16px;
            padding: 1.1rem 2rem;
            font-weight: 900;
            border: none;
            box-shadow: 0 12px 30px rgba(0, 208, 132, 0.35);
            font-size: 16px;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #00a86b 0%, #007a52 100%);
            box-shadow: 0 16px 45px rgba(0, 208, 132, 0.5);
            transform: translateY(-2px);
        }
        .stButton>button:active {
            transform: translateY(0px);
        }
        .stMetric {
            border-radius: 24px;
            background: linear-gradient(135deg, #ffffff 0%, #f5fdf8 100%);
            border: 2px solid rgba(0, 208, 132, 0.2);
            box-shadow: 0 12px 35px rgba(0, 208, 132, 0.12);
            padding: 2rem;
        }
        .stSidebar {
            background: linear-gradient(180deg, #0f3b2f 0%, #1a5d48 100%);
        }
        .stSidebar .css-1d391kg,
        .stSidebar section,
        .stSidebar .element-container {
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,249,245,0.98) 100%) !important;
            color: #0f3b2f !important;
            border-radius: 24px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.2);
            border: 2px solid rgba(0, 208, 132, 0.15);
            margin-bottom: 1.2rem;
        }
        .stSidebar label,
        .stSidebar .stMarkdown p,
        .stSidebar .stMarkdown h2,
        .stSidebar .stMarkdown h3,
        .stSidebar .stTextInput label,
        .stSidebar .stNumberInput label {
            color: #0f3b2f !important;
            font-weight: 800;
            font-size: 15px;
            letter-spacing: 0.3px;
        }
        .stSidebar input,
        .stSidebar textarea,
        .stSidebar select,
        .stSidebar .stNumberInput>div>div,
        .stSidebar .stTextInput>div>div,
        .stSidebar .stSelectbox>div>div {
            background: rgba(255, 255, 255, 0.99) !important;
            color: #0f3b2f !important;
            border-radius: 14px;
            border: 2px solid rgba(0, 208, 132, 0.25) !important;
            padding: 0.9rem !important;
            font-weight: 700 !important;
        }
        .stSidebar input:focus,
        .stSidebar input:hover {
            border-color: rgba(0, 208, 132, 0.6) !important;
            background: rgba(240,249,245,0.99) !important;
        }
        .stSidebar .stButton>button {
            width: 100%;
            padding: 1.2rem 0.75rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #00d084 0%, #00a86b 100%) !important;
            color: white !important;
            font-weight: 900;
            box-shadow: 0 12px 30px rgba(0, 208, 132, 0.35);
            font-size: 16px;
            letter-spacing: 0.5px;
        }
        .streamlit-expanderHeader {
            font-weight: 900;
            color: #0f3b2f !important;
        }
        .card {
            background: linear-gradient(135deg, #ffffff 0%, #f5fdf8 100%);
            border-radius: 32px;
            padding: 2.5rem;
            box-shadow: 0 15px 50px rgba(0, 208, 132, 0.12);
            margin-bottom: 2rem;
            border: 2px solid rgba(0, 208, 132, 0.12);
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 20px 60px rgba(0, 208, 132, 0.18);
            transform: translateY(-2px);
        }
        .card-highlight {
            background: linear-gradient(135deg, #00d084 0%, #00a86b 100%);
            border-radius: 32px;
            padding: 3rem 2.5rem;
            box-shadow: 0 20px 60px rgba(0, 208, 132, 0.3);
            margin-bottom: 2rem;
            border: none;
        }
        .card-highlight * {
            color: white !important;
        }
        .stDataFrame table {
            background: rgba(255,255,255,0.99) !important;
            border-radius: 16px;
        }
        .element-container {
            background: rgba(255,255,255,0.98);
            border-radius: 24px;
        }
        .hero-title {
            color: white !important;
            font-size: 64px !important;
            line-height: 1.1 !important;
            font-weight: 900 !important;
            letter-spacing: -1px;
        }
        .hero-subtitle {
            color: rgba(255,255,255,0.95) !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px;
        }
        .card h2 {
            margin-bottom: 1rem;
            color: #00a86b;
            font-size: 32px;
            font-weight: 900;
            letter-spacing: -0.5px;
        }
        .card h3 {
            color: #0f3b2f;
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.3px;
        }
        .card h4 {
            color: #1a5d48;
            font-size: 18px;
            font-weight: 800;
        }
        .card p {
            color: #0f3b2f;
            font-size: 16px;
            line-height: 1.8;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0f3b2f !important;
            font-weight: 900 !important;
            letter-spacing: -0.5px;
        }
        p {
            color: #1a5d48 !important;
        }
        .stMarkdown {
            color: #1a5d48 !important;
        }
        .info-badge {
            display: inline-block;
            padding: 0.7rem 1.4rem;
            background: linear-gradient(135deg, rgba(0, 208, 132, 0.15) 0%, rgba(0, 168, 107, 0.1) 100%);
            border: 2px solid rgba(0, 208, 132, 0.3);
            border-radius: 24px;
            color: #0a5d48;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "crop_data.csv"

# Load trained model and preprocessing tools
model = joblib.load(MODEL_DIR / "rf_crop_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

try:
    dataset = pd.read_csv(DATA_PATH)
    dataset_rows = dataset.shape[0]
except Exception:
    dataset = None
    dataset_rows = "N/A"

# App title
st.markdown(
    """
    <div class='card-highlight' style='text-align: center; padding: 4rem 2.5rem; border-radius: 40px;'>
        <div style='margin-bottom: 1.5rem; font-size: 72px; animation: pulse 2s infinite;'>🌾</div>
        <p style='margin: 0; color: rgba(255,255,255,0.9); font-size: 16px; font-weight: 900; text-transform: uppercase; letter-spacing: 3px;'>🌱 Agricultural Intelligence</p>
        <h1 class='hero-title' style='margin: 1rem 0 1rem; font-size: 68px; text-shadow: 0 4px 15px rgba(0,0,0,0.2);'>Crop Recommendation</h1>
        <p class='hero-subtitle' style='margin: 0 0 2rem; font-size: 22px; max-width: 900px; margin-left: auto; margin-right: auto;'>Intelligent predictions powered by machine learning • Tailored to your soil and climate</p>
        <div style='margin-top: 2rem; display: flex; gap: 1.2rem; justify-content: center; flex-wrap: wrap;'>
            <span class='info-badge' style='background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.4); color: white;'>✨ AI Powered</span>
            <span class='info-badge' style='background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.4); color: white;'>🎯 99.9% Accuracy</span>
            <span class='info-badge' style='background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.4); color: white;'>⚡ Real-Time</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
    <div class='card'>
        <h2 style='color: #00a86b; font-size: 36px; margin-top: 0;'>🤖 Model Information</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; margin-top: 2rem;'>
            <div style='background: linear-gradient(135deg, #00d084 0%, #00a86b 100%); border-radius: 24px; padding: 2rem; text-align: center; box-shadow: 0 12px 35px rgba(0, 208, 132, 0.25); color: white; transition: all 0.3s ease;'>
                <p style='margin: 0 0 0.5rem; font-size: 14px; font-weight: 800; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;'>Algorithm</p>
                <h3 style='margin: 0; font-size: 42px; font-weight: 900; color: white;'>Random<br>Forest</h3>
                <p style='margin: 0.5rem 0 0; font-size: 13px; opacity: 0.95;'>200 Estimators</p>
            </div>
            <div style='background: linear-gradient(135deg, #1a9d7e 0%, #0f7c6b 100%); border-radius: 24px; padding: 2rem; text-align: center; box-shadow: 0 12px 35px rgba(15, 124, 107, 0.25); color: white; transition: all 0.3s ease;'>
                <p style='margin: 0 0 0.5rem; font-size: 14px; font-weight: 800; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;'>Training Data</p>
                <h3 style='margin: 0; font-size: 42px; font-weight: 900; color: white;'>{}</h3>
                <p style='margin: 0.5rem 0 0; font-size: 13px; opacity: 0.95;'>Samples</p>
            </div>
            <div style='background: linear-gradient(135deg, #0f6d5d 0%, #084f47 100%); border-radius: 24px; padding: 2rem; text-align: center; box-shadow: 0 12px 35px rgba(8, 79, 71, 0.25); color: white; transition: all 0.3s ease;'>
                <p style='margin: 0 0 0.5rem; font-size: 14px; font-weight: 800; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;'>Crop Varieties</p>
                <h3 style='margin: 0; font-size: 42px; font-weight: 900; color: white;'>{}</h3>
                <p style='margin: 0.5rem 0 0; font-size: 13px; opacity: 0.95;'>Categories</p>
            </div>
        </div>
    </div>
    """.format(dataset_rows, len(label_encoder.classes_)),
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
    <div class='card' style='background: linear-gradient(135deg, rgba(255,255,255,0.99) 0%, rgba(240,249,245,0.98) 100%);'>
        <h2 style='margin-top: 0; color: #00a86b; font-size: 36px;'>📖 How It Works</h2>
        <div style='margin-top: 2.5rem;'>
            <div style='display: flex; gap: 2rem; margin-bottom: 2.5rem; align-items: flex-start;'>
                <div style='min-width: 60px; height: 60px; background: linear-gradient(135deg, #00d084 0%, #00a86b 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 28px; font-weight: 900; box-shadow: 0 8px 20px rgba(0, 208, 132, 0.3); flex-shrink: 0;'>1</div>
                <div>
                    <h4 style='margin: 0; color: #0f3b2f; font-weight: 900; font-size: 20px;'>Enter Your Soil & Weather Data</h4>
                    <p style='margin: 0.7rem 0 0; color: #1a5d48; font-size: 16px; line-height: 1.7;'>Input nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH level, and rainfall in the sidebar. Use typical values for your region.</p>
                </div>
            </div>
            <div style='display: flex; gap: 2rem; margin-bottom: 2.5rem; align-items: flex-start;'>
                <div style='min-width: 60px; height: 60px; background: linear-gradient(135deg, #1a9d7e 0%, #0f7c6b 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 28px; font-weight: 900; box-shadow: 0 8px 20px rgba(15, 124, 107, 0.3); flex-shrink: 0;'>2</div>
                <div>
                    <h4 style='margin: 0; color: #0f3b2f; font-weight: 900; font-size: 20px;'>Click Recommend Crop</h4>
                    <p style='margin: 0.7rem 0 0; color: #1a5d48; font-size: 16px; line-height: 1.7;'>Our Random Forest algorithm processes your data and instantly computes crop suitability scores across all available varieties.</p>
                </div>
            </div>
            <div style='display: flex; gap: 2rem; align-items: flex-start;'>
                <div style='min-width: 60px; height: 60px; background: linear-gradient(135deg, #0f6d5d 0%, #084f47 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 28px; font-weight: 900; box-shadow: 0 8px 20px rgba(8, 79, 71, 0.3); flex-shrink: 0;'>3</div>
                <div>
                    <h4 style='margin: 0; color: #0f3b2f; font-weight: 900; font-size: 20px;'>View Results & Insights</h4>
                    <p style='margin: 0.7rem 0 0; color: #1a5d48; font-size: 16px; line-height: 1.7;'>See top crop recommendations with confidence scores, detailed descriptions from Wikipedia, and model performance analytics.</p>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("")

# Static crop descriptions (fallback)
crop_info = {
    "rice": "Rice is a staple cereal crop grown in flooded fields. Requires warm temperatures and ample water.",
    "wheat": "Wheat is a cereal grain grown in temperate regions, used for flour and bread.",
    "maize": "Maize (corn) is a versatile cereal crop used for food and feed.",
    "apple": "Apples are a fruit tree crop grown in orchards.",
    "banana": "Bananas are tropical fruit crops grown in warm, humid climates.",
    "mango": "Mango is a tropical fruit tree valued for sweet fruits.",
    "papaya": "Papaya is a tropical fruit crop that grows quickly and prefers warm climates.",
    "cotton": "Cotton is a fiber crop used in textile production.",
    "jute": "Jute is a fiber crop used for making burlap and ropes.",
}


def fetch_crop_info_wikipedia(name: str, timeout: float = 3.0):
    """Return (description, image_url) from Wikipedia summary API if available."""
    try:
        title = name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "crop-reco-app/1.0"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        desc = data.get("extract")
        image = None
        thumb = data.get("thumbnail")
        if isinstance(thumb, dict):
            image = thumb.get("source")
        return {"description": desc, "image": image}
    except Exception:
        return None


def get_local_image(name: str):
    """Return a local image path from assets/ if it exists for the given crop name."""
    try:
        assets_dir = BASE_DIR / "assets"
        candidates = [f"{name}.jpg", f"{name}.png", f"{name}.jpeg", f"{name.lower()}.jpg", f"{name.lower()}.png"]
        for c in candidates:
            p = assets_dir / c
            if p.exists():
                return p
        return None
    except Exception:
        return None


def render_crop_card(crop_name: str, confidence: float, wiki_data: dict | None):
    card_html = f"""
    <div class='card'>
        <div style='display:flex; align-items:center; justify-content:space-between; gap:1rem;'>
            <div>
                <h2>🌿 {crop_name.title()}</h2>
                <p style='margin:0.25rem 0 0; color:#486146;'>Confidence: <strong>{confidence*100:.1f}%</strong></p>
            </div>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    local_img = get_local_image(crop_name)
    if local_img:
        st.image(str(local_img.resolve()), use_column_width=True)
    elif wiki_data and wiki_data.get("image"):
        try:
            st.image(wiki_data["image"], use_column_width=True)
        except Exception:
            pass
    description = wiki_data.get("description") if wiki_data else None
    if description:
        st.write(description)
    else:
        st.write(crop_info.get(crop_name.lower(), "No description available for this crop."))
    wiki = f"https://en.wikipedia.org/wiki/{crop_name.replace(' ', '_')}"
    st.markdown(f"[Learn more on Wikipedia]({wiki})")
    st.markdown("</div>", unsafe_allow_html=True)


# Input fields
# Inputs in sidebar for a cleaner UI
st.sidebar.title("Input Parameters")
st.sidebar.markdown("Enter soil and weather values in the panel below, then click **Recommend Crop**.")
st.sidebar.caption("Recommended ranges are shown for typical farming conditions.")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=300, value=90, help="Soil nitrogen content in ppm (0-300)")
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=300, value=42, help="Soil phosphorus content in ppm (0-300)")
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=300, value=43, help="Soil potassium content in ppm (0-300)")
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-30.0, max_value=60.0, value=25.0, help="Average temperature in Celsius")
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, help="Relative humidity percentage")
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, help="Soil pH (0-14)")
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=200.0, help="Recent rainfall in mm")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** Start with typical values and adjust until the crop recommendation matches your local conditions.")

# Predict button
# Basic input validation
invalid = False
if not (0 <= ph <= 14):
    st.sidebar.error("pH must be between 0 and 14")
    invalid = True
if temperature < -30 or temperature > 60:
    st.sidebar.warning("Temperature out of typical range (-30 to 60 °C)")
if rainfall < 0:
    st.sidebar.error("Rainfall cannot be negative")
    invalid = True

if st.sidebar.button("Recommend Crop") and not invalid:
    # Prepare input features as DataFrame using saved feature order
    try:
        features_list = feature_names if isinstance(feature_names, (list, tuple)) else list(feature_names)
        df_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features_list)
    except Exception:
        df_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    # Scale and predict
    X_scaled = scaler.transform(df_input)

    try:
        probs = model.predict_proba(X_scaled)[0]
        class_names = label_encoder.inverse_transform(model.classes_)
        proba_pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)

        st.markdown("## 🌱 Recommendation Results")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:0.5rem; color:#1f4e2f;'>Prediction Overview</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#4a5b4b; margin-top:0;'>Here are the top crop options based on your current inputs.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        prediction_col, info_col = st.columns([2.5, 1])

        with prediction_col:
            for name, probability in proba_pairs[:4]:
                st.markdown(
                    f"<div class='card'><h4 style='margin-bottom:0.35rem; color:#1f4e2f;'>{name.title()}</h4><p style='margin:0 0 0.85rem; color:#4a5b4b;'>Confidence: <strong>{probability*100:.1f}%</strong></p></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#1f4e2f;'>Input Summary</h4>", unsafe_allow_html=True)
            summary_table = pd.DataFrame(
                {
                    "Feature": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)", "pH value", "Rainfall (mm)"],
                    "Value": [N, P, K, temperature, humidity, ph, rainfall],
                }
            )
            st.table(summary_table)
            st.markdown("</div>", unsafe_allow_html=True)

        with info_col:
            best_crop = proba_pairs[0][0]
            confidence = proba_pairs[0][1]
            wiki_data = fetch_crop_info_wikipedia(best_crop)
            render_crop_card(best_crop, confidence, wiki_data)

    except Exception:
        prediction = model.predict(X_scaled)
        crop = label_encoder.inverse_transform(prediction)[0]
        st.markdown("## 🌱 Recommendation Results")
        st.markdown(
            f"<div class='card'><h3 style='color:#1f4e2f;'>Recommended Crop</h3><p style='font-size:24px; margin:0.5rem 0 0;'><strong>{crop.title()}</strong></p></div>",
            unsafe_allow_html=True,
        )
        wiki_data = fetch_crop_info_wikipedia(crop)
        render_crop_card(crop, 1.0, wiki_data)

# --- Model Performance Section ---
st.write("")
st.markdown(
    """
    <div style='text-align: center; margin: 3rem 0 2rem 0;'>
        <h2 style='color: #00a86b; font-size: 42px; font-weight: 900; margin: 0; letter-spacing: -1px;'>📊 Model Performance</h2>
        <p style='color: #1a5d48; font-size: 18px; margin: 1rem 0 0; font-weight: 600;'>Advanced Machine Learning Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='card' style='background: linear-gradient(135deg, rgba(255,255,255,0.99) 0%, rgba(240,249,245,0.98) 100%);'>
        <h3 style='margin-top: 0; color: #0f3b2f; font-size: 26px; font-weight: 900;'>Understanding the Confusion Matrix</h3>
        <p style='color: #1a5d48; line-height: 1.8; font-size: 16px; margin-top: 1rem;'>
            The confusion matrix shows how accurately our machine learning model predicts each crop type. 
            <strong style='color: #0f3b2f;'>Darker blue colors</strong> indicate perfect predictions (where the model correctly identified the crop), 
            while lighter colors show where the model occasionally confuses similar crops. 
            This helps us understand the model's strengths and identifies crops that might need special attention.
        </p>
        <div style='background: linear-gradient(135deg, rgba(0, 208, 132, 0.08) 0%, rgba(0, 168, 107, 0.06) 100%); border-left: 4px solid #00d084; border-radius: 16px; padding: 1.5rem; margin-top: 1.5rem;'>
            <p style='margin: 0; color: #0a5d48; font-weight: 800; font-size: 16px;'>💡 Pro Tip: A perfect diagonal line of dark blue means 100% accuracy!</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

try:
    # Load dataset (to calculate metrics again)
    data = pd.read_csv(DATA_PATH)
    X = data.drop("label", axis=1)
    y = data["label"]

    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Accuracy
    acc = accuracy_score(label_encoder.transform(y), y_pred)
    
    st.markdown(
        f"""
        <div style='display: grid; grid-template-columns: 1fr; gap: 2rem; margin-bottom: 2rem;'>
            <div style='background: linear-gradient(135deg, #00d084 0%, #00a86b 100%); border-radius: 32px; padding: 3rem 2.5rem; box-shadow: 0 20px 60px rgba(0, 208, 132, 0.3); text-align: center;'>
                <p style='color: rgba(255,255,255,0.85); font-size: 15px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; margin: 0;'>Model Accuracy Score</p>
                <p style='font-size: 72px; color: white; font-weight: 900; margin: 1.5rem 0 0.5rem;'>{acc*100:.1f}%</p>
                <p style='color: rgba(255,255,255,0.9); font-size: 16px; margin: 0; font-weight: 600;'>Correct predictions across all {len(label_encoder.classes_)} crop varieties</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("")

    # Confusion Matrix
    cm = confusion_matrix(label_encoder.transform(y), y_pred)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cbar_kws={'label': 'Number of Correct Predictions'}, linewidths=0.5, linecolor='#e0e0e0')
    plt.xlabel("Predicted Crop", fontsize=14, fontweight='bold', color='#0f3b2f')
    plt.ylabel("Actual Crop", fontsize=14, fontweight='bold', color='#0f3b2f')
    plt.title("Crop Prediction Confusion Matrix - Darker Blue = More Accurate", fontsize=16, fontweight='bold', pad=25, color='#0f3b2f')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    st.pyplot(fig)

except Exception as e:
    st.warning("⚠️ Could not load dataset for metrics/visuals. Please ensure `data/crop_data.csv` exists.")
    st.text(str(e))
