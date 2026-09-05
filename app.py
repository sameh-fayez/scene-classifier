"""
Streamlit app — Scene Atlas (Natural Scene Image Classifier)
Computer Vision Project: Multi-Class Image Classification
(buildings, forest, glacier, mountain, sea, street)

Run locally:
    pip install streamlit tensorflow pillow numpy requests
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt + .streamlit/config.toml to a GitHub repo
       (final_model.keras itself does NOT need to be in the repo — it's downloaded
       automatically at startup from MODEL_URL below).
    2. Go to https://share.streamlit.io, connect the repo, set app.py as the entry point.
"""

import os
import numpy as np
import requests
import streamlit as st
from PIL import Image
from tensorflow import keras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_URL = "https://huggingface.co/samehfayez/scene-classifier-model/resolve/main/final_model.keras"
MODEL_PATH = "final_model.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Scene Atlas", page_icon="◆", layout="centered")

# ---------------------------------------------------------------------------
# Styling — a quiet, editorial "atelier" look: warm near-black, a single
# champagne-gold accent, glass panels, and a serif display for the result.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Manrope:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

    .stApp {
        background: radial-gradient(120% 140% at 15% 0%, #1A1D22 0%, #0B0D10 55%, #08090B 100%);
    }

    .atlas-header h1 {
        font-family: 'Cormorant Garamond', serif; font-weight: 600; font-style: italic;
        font-size: 2.6rem; color: #F3F1EC; margin: 0 0 0.2rem 0; letter-spacing: 0.3px;
    }
    .atlas-sub { color: #8B887F; font-size: 0.97rem; margin-bottom: 1.8rem; max-width: 48ch; line-height: 1.5; }
    .atlas-sub b { color: #D4B98C; font-weight: 500; }

    .glass-panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.7rem 1.8rem;
        margin-top: 0.5rem;
        backdrop-filter: blur(18px);
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
    }
    .glass-panel.empty { text-align: center; color: #6C6960; font-size: 0.95rem; padding: 2.4rem 1.8rem; }

    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.4);
    }

    .result-label {
        color: #7A776E; font-size: 0.8rem; letter-spacing: 0.4px; margin-bottom: 0.3rem;
    }
    .result-name {
        font-family: 'Cormorant Garamond', serif; font-weight: 600; font-size: 2.7rem;
        color: #F3F1EC; line-height: 1.05; margin-bottom: 1.1rem;
    }

    .meter-track {
        height: 5px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .meter-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, #9C824F 0%, #D4B98C 60%, #EAD9B8 100%);
        box-shadow: 0 0 14px rgba(212,185,140,0.45);
    }
    .meter-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }
    .meter-pct { font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; color: #D4B98C; }
    .meter-caption { color: #7A776E; font-size: 0.85rem; margin-bottom: 1.5rem; }

    .breakdown-title {
        color: #7A776E; font-size: 0.8rem; letter-spacing: 0.4px;
        margin-bottom: 0.7rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1.1rem;
    }
    .bar-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.5rem; }
    .bar-label { width: 6.2rem; font-size: 0.88rem; color: #C7C4BB; flex-shrink: 0; }
    .bar-track { flex: 1; height: 4px; background: rgba(255,255,255,0.07); border-radius: 2px; overflow: hidden; }
    .bar-fill { height: 100%; background: rgba(255,255,255,0.22); border-radius: 2px; }
    .bar-fill.top { background: linear-gradient(90deg, #9C824F, #D4B98C); }
    .bar-pct { width: 3rem; font-size: 0.83rem; color: #7A776E; text-align: right; flex-shrink: 0; }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 14px;
    }

    footer, #MainMenu { visibility: hidden; }
    .atlas-footer {
        color: #4E4C46; font-size: 0.78rem; margin-top: 2.4rem;
        border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def download_model(url: str, path: str, min_valid_size: int = 1_000_000):
    """Downloads the model file, skipping only if a file already on disk looks like a
    real model (not a leftover empty/corrupt file from an earlier deploy attempt)."""
    if os.path.exists(path) and os.path.getsize(path) >= min_valid_size:
        return
    with st.spinner("Fetching the model (first run only)..."):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    if os.path.getsize(path) < min_valid_size:
        raise RuntimeError(
            f"Downloaded file is only {os.path.getsize(path)} bytes — the URL likely "
            f"returned an error page instead of the model file. Check that MODEL_URL "
            f"is a direct download link."
        )


@st.cache_resource
def load_model(path: str):
    """Loaded once per server session, not on every prediction."""
    download_model(MODEL_URL, path)
    return keras.models.load_model(path)


def predict(image: Image.Image, model) -> dict:
    """Runs the same preprocessing the training notebook applies at inference:
    resize to IMG_SIZE and feed raw 0-255 RGB pixels — the model's own input
    layers (Rescaling / preprocess_input) handle normalization internally."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img), axis=0).astype("float32")

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": dict(zip(CLASS_NAMES, probs.tolist())),
    }


def confidence_caption(confidence: float) -> str:
    if confidence >= 0.85:
        return "Confident match"
    if confidence >= 0.60:
        return "Fairly confident"
    return "Uncertain — the scene may mix categories"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="atlas-header"><h1>Scene Atlas</h1></div>
    <div class="atlas-sub">Upload a photograph and it's classified into one of six
    natural-scene categories: <b>buildings, forest, glacier, mountain, sea, street</b>.</div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not download or load the model from `{MODEL_URL}`.\n\nDetails: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a photograph", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.1], gap="large")
    with col1:
        st.image(image, use_column_width=True)

    with st.spinner("Identifying scene..."):
        result = predict(image, model)

    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda kv: kv[1], reverse=True
    )
    bars_html = "".join([
        f'<div class="bar-row">'
        f'<div class="bar-label">{cls.capitalize()}</div>'
        f'<div class="bar-track"><div class="bar-fill {"top" if i == 0 else ""}" style="width:{prob*100:.1f}%;"></div></div>'
        f'<div class="bar-pct">{prob*100:.1f}%</div>'
        f'</div>'
        for i, (cls, prob) in enumerate(sorted_probs)
    ])

    with col2:
        st.markdown(
            f"""
            <div class="glass-panel">
                <div class="result-label">Identified as</div>
                <div class="result-name">{result['predicted_class'].capitalize()}</div>
                <div class="meter-row">
                    <span></span><span class="meter-pct">{result['confidence']*100:.1f}%</span>
                </div>
                <div class="meter-track">
                    <div class="meter-fill" style="width:{result['confidence']*100:.1f}%;"></div>
                </div>
                <div class="meter-caption">{confidence_caption(result['confidence'])}</div>
                <div class="breakdown-title">All categories</div>
                {bars_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="glass-panel empty">Drop a photograph above to see it identified.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="atlas-footer">
    Model: MobileNetV2 transfer-learning classifier trained on the Intel Image
    Classification dataset (test accuracy 91.07%, macro F1 0.9122).
    </div>
    """,
    unsafe_allow_html=True,
)
