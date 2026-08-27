"""
Train the weather condition classifier.

Usage:
    python src/train.py
"""

import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config
from data_loader import get_data_generators
from model import build_model


def plot_history(history_list, save_path):
    """Combine multiple History objects (head phase + fine-tune phase) and plot."""
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_list:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tf.random.set_seed(config.RANDOM_SEED)

    print("Loading data generators...")
    train_gen, val_gen, test_gen = get_data_generators()

    # Save class index mapping for later use (predict.py / app.py)
    with open(config.CLASS_INDICES_PATH, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Class indices: {train_gen.class_indices}")

    callbacks = [
        ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # ---------------- Phase 1: train head with frozen backbone ----------------
    print("\n=== Phase 1: Training classification head (backbone frozen) ===")
    model, base_model = build_model(fine_tune=False)
    model.compile(
        optimizer=Adam(learning_rate=config.HEAD_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.HEAD_EPOCHS,
        callbacks=callbacks,
    )

    # ---------------- Phase 2: fine-tune top backbone layers -------------------
    print("\n=== Phase 2: Fine-tuning top backbone layers ===")
    for layer in base_model.layers[: config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    for layer in base_model.layers[config.FINE_TUNE_AT_LAYER :]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    # ---------------- Save final model & plots ----------------
    model.save(config.FINAL_MODEL_PATH)
    print(f"Saved final model to {config.FINAL_MODEL_PATH}")
    print(f"Best checkpoint saved to {config.BEST_MODEL_PATH}")

    plot_history([history_head, history_fine], config.HISTORY_PLOT_PATH)

    # ---------------- Quick test set evaluation ----------------
    print("\n=== Evaluating on test set ===")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
