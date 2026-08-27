"""
Predict the weather condition for a single image.

Usage:
    python src/predict.py --image path/to/photo.jpg
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import config


def load_class_labels():
    """Load class -> index mapping saved during training, else fall back to config."""
    if os.path.exists(config.CLASS_INDICES_PATH):
        with open(config.CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # invert: index -> class name
        return {v: k for k, v in class_indices.items()}
    return {i: name for i, name in enumerate(config.CLASS_NAMES)}


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def predict(img_path, model_path=config.FINAL_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with src/train.py"
        )
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = tf.keras.models.load_model(model_path)
    idx_to_class = load_class_labels()

    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    print(f"\nPredicted class : {predicted_class.capitalize()}")
    print(f"Confidence      : {confidence:.2f}%")
    print("\nClass probabilities:")
    for idx, prob in enumerate(predictions):
        print(f"  {idx_to_class[idx]:<10}: {prob * 100:6.2f}%")

    return predicted_class, confidence, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument(
        "--model", type=str, default=config.FINAL_MODEL_PATH, help="Path to trained model"
    )
    args = parser.parse_args()

    predict(args.image, args.model)
