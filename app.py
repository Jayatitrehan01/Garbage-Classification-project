import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 128
MODEL_PATH = "trashvision_model.h5"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = sorted(os.listdir(os.path.join(SCRIPT_DIR, "dataset")))

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_trash_model():
    return load_model(MODEL_PATH)

model = load_trash_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🗑️ Garbage Classifier - TrashVision")
st.write("Upload an image to classify it into one of the garbage categories.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Show result
    st.markdown(f"### 🧠 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
