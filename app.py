import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("😷 Face Mask Detection App (Softmax Based)")
st.markdown("Upload an image by **drag & drop** or click to browse.")

# Load model
@st.cache_resource
def load_mask_model():
    return load_model("mask_img_analyzer.h5", compile=False)

model = load_mask_model()

# Prediction function
def predict_mask(image_np):
    image_resized = cv2.resize(image_np, (128, 128))  # Resize to model input
    image_scaled = image_resized / 255.0              # Normalize
    image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])  # Add batch dimension

    prediction = model.predict(image_reshaped)
    predicted_label = np.argmax(prediction, axis=1)[0]

    label = "Mask" if predicted_label == 1 else "No Mask"
    confidence = float(np.max(prediction))  # Maximum probability score
    return label, confidence

# Upload section
uploaded_file = st.file_uploader("📁 Upload or Drop your Image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    label, confidence = predict_mask(image_np)
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"🧪 Confidence Score: `{confidence*100:.2f}%`")
