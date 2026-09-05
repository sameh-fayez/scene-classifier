"""
Streamlit app — Natural Scene Image Classifier
Computer Vision Project: Multi-Class Image Classification
(buildings, forest, glacier, mountain, sea, street)

Run locally:
    pip install streamlit tensorflow pillow numpy
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + final_model.keras to a GitHub repo.
    2. Go to https://share.streamlit.io, connect the repo, set app.py as the entry point.
    3. Make sure final_model.keras is committed to the repo (or hosted somewhere the app
       can download it from at startup) so MODEL_PATH below resolves correctly.
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "final_model.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Scene Classifier", page_icon="🏞️", layout="centered")


@st.cache_resource
def load_model(path: str):
    """Loaded once per server session, not on every prediction."""
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
        f"Could not load the model from `{MODEL_PATH}`. Make sure `final_model.keras` "
        f"(produced by the training notebook, Section 19) is in the same folder as this "
        f"app.\n\nDetails: {e}"
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
