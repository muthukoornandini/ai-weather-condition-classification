"""
Streamlit web app for the AI-Based Weather Condition Classification System.

Usage:
    streamlit run app.py
"""

import json
import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config  # noqa: E402

st.set_page_config(page_title="Weather Classifier", page_icon="🌦️", layout="centered")

WEATHER_EMOJI = {
    "cloudy": "☁️",
    "rain": "🌧️",
    "shine": "☀️",
    "sunrise": "🌅",
}


@st.cache_resource
def load_model_and_labels():
    model_path = config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        return None, None

    model = tf.keras.models.load_model(model_path)

    if os.path.exists(config.CLASS_INDICES_PATH):
        with open(config.CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        idx_to_class = {i: name for i, name in enumerate(config.CLASS_NAMES)}

    return model, idx_to_class


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🌦️ AI-Based Weather Condition Classification")
    st.write(
        "Upload a photo and the model will predict the weather condition: "
        "**Cloudy**, **Rain**, **Shine**, or **Sunrise**."
    )

    model, idx_to_class = load_model_and_labels()

    if model is None:
        st.error(
            "No trained model found. Please run `python src/train.py` first "
            "to train and save a model, then relaunch this app."
        )
        return

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing weather condition..."):
            input_arr = preprocess(img)
            predictions = model.predict(input_arr)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = idx_to_class[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        emoji = WEATHER_EMOJI.get(predicted_class, "🌈")

        st.success(f"{emoji} **Prediction: {predicted_class.capitalize()}**")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Class probabilities")
        prob_dict = {
            idx_to_class[i].capitalize(): float(p) for i, p in enumerate(predictions)
        }
        st.bar_chart(prob_dict)


if __name__ == "__main__":
    main()
