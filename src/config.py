"""
Central configuration for the Weather Condition Classification project.
Edit values here rather than hard-coding them elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "weather_model_best.h5")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "weather_model_final.h5")
HISTORY_PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Update these if your dataset uses different class folder names.
CLASS_NAMES = ["cloudy", "rain", "shine", "sunrise"]
NUM_CLASSES = len(CLASS_NAMES)

# Train / val / test split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 32

# Phase 1: train the classification head only (backbone frozen)
HEAD_EPOCHS = 15
HEAD_LEARNING_RATE = 1e-3

# Phase 2: fine-tune the top of the backbone
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT_LAYER = 100  # unfreeze layers from this index onward

DROPOUT_1 = 0.4
DROPOUT_2 = 0.3
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 128

RANDOM_SEED = 42
