"""
Evaluate the trained model on the test set:
- Overall accuracy
- Per-class precision / recall / F1
- Confusion matrix (saved as an image)

Usage:
    python src/evaluate.py [--model models/weather_model_final.h5]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import get_data_generators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=config.FINAL_MODEL_PATH,
        help="Path to a trained .h5 model file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found at {args.model}. Train it first with src/train.py"
        )

    print(f"Loading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model)

    print("Loading test data...")
    _, _, test_gen = get_data_generators()

    print("Running predictions on test set...")
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    class_labels = list(test_gen.class_indices.keys())

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Weather Classification")
    plt.tight_layout()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    print(f"\nSaved confusion matrix to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
