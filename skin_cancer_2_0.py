import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Page Configuration
st.set_page_config(
    page_title="Skin Cancer Detection (HAM10000)",
    page_icon="🩺",
    layout="centered"
)

st.title("Skin Cancer Detection (HAM10000)")
st.write("Upload a dermatoscopic skin lesion image to classify it across the 7 HAM10000 diagnostic categories.")

# HAM10000 Class Labels (alphabetical order used by Keras/ImageDataGenerator)
CLASS_NAMES = [
    "Actinic keratoses and intraepithelial carcinoma (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vasc)"
]

# Update this filename to match the exact .h5 filename in your GitHub repo
MODEL_PATH = "skin_cancer_model.h5"


def build_fallback_architecture(input_shape=(28, 28, 3), num_classes=7):
    """
    Exact layer structure extracted from your .h5 file:
    3x [Conv2D -> BatchNorm -> MaxPooling2D -> Dropout] -> Flatten -> Dense -> BatchNorm -> Dropout -> Dense
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d"),
        layers.BatchNormalization(name="batch_normalization"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d"),
        layers.Dropout(0.25, name="dropout"),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_1"),
        layers.BatchNormalization(name="batch_normalization_1"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        layers.Dropout(0.25, name="dropout_1"),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_2"),
        layers.BatchNormalization(name="batch_normalization_2"),
        layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        layers.Dropout(0.25, name="dropout_2"),

        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.BatchNormalization(name="batch_normalization_3"),
        layers.Dropout(0.5, name="dropout_3"),
        layers.Dense(num_classes, activation="softmax", name="dense_1")
    ])
    return model


@st.cache_resource
def load_trained_model():
    # Automatically locate any .h5 or .keras file in the repo if MODEL_PATH isn't exact
    model_file = MODEL_PATH
    if not os.path.exists(model_file):
        h5_files = [f for f in os.listdir(".") if f.endswith((".h5", ".keras"))]
        if not h5_files:
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Upload your .h5 file to the root of your GitHub repository."
            )
        model_file = h5_files[0]

    # Try standard loading without compiling (avoids Keras 2 -> Keras 3 optimizer ValueError)
    try:
        return tf.keras.models.load_model(model_file, compile=False)
    except Exception:
        # Fallback: build matching architecture and load weights directly
        model = build_fallback_architecture()
        model.load_weights(model_file)
        return model


# Load Model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Dynamically detect expected image dimensions (e.g., 28x28, 64x64, or 224x224) from the model
input_shape = model.input_shape
img_height = input_shape[1] if input_shape[1] else 28
img_width = input_shape[2] if input_shape[2] else 28

# File Uploader & Inference
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image to match model input
    img_resized = image.resize((img_width, img_height))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    if st.button("Analyze Lesion"):
        with st.spinner("Running classification..."):
            predictions = model.predict(img_batch)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx]) * 100

        st.subheader(f"Prediction: {CLASS_NAMES[top_idx]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.write("### Class Probabilities")
        for label, prob in sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True):
            st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")

        st.warning("Disclaimer: This tool is for educational and research purposes only and is not a medical diagnosis.")
