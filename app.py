"""
Streamlit app — Natural Scene Image Classifier
Computer Vision Project: Multi-Class Image Classification
(buildings, forest, glacier, mountain, sea, street)

Run locally:
    pip install streamlit tensorflow pillow numpy requests
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt to a GitHub repo (final_model.keras itself does
       NOT need to be in the repo — it's downloaded automatically at startup from
       MODEL_URL below, since large model files often fail to upload via GitHub's
       browser drag-and-drop).
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

st.set_page_config(page_title="Scene Classifier", page_icon="🏞️", layout="centered")


def download_model(url: str, path: str):
    """Downloads the model file once (skipped if it already exists on disk from a
    previous run in this same container)."""
    if os.path.exists(path):
        return
    with st.spinner("Downloading model (first run only)..."):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏞️ Natural Scene Classifier")
st.write(
    "Upload a photo and the model will classify it into one of six natural-scene "
    "categories: **buildings, forest, glacier, mountain, sea, street**."
)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        f"Could not download or load the model from `{MODEL_URL}`.\n\nDetails: {e}"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        result = predict(image, model)

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", result["predicted_class"].capitalize())
        st.metric("Confidence", f"{result['confidence']:.1%}")

        st.subheader("All class probabilities")
        probs_sorted = dict(
            sorted(result["all_probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        )
        st.bar_chart(probs_sorted)
else:
    st.info("👆 Upload an image to get a prediction.")

st.divider()
st.caption(
    "Model: MobileNetV2 transfer-learning classifier trained on the Intel Image "
    "Classification dataset (test accuracy 91.07%, macro F1 0.9122). "
    "See the accompanying notebook for the full training pipeline."
)
