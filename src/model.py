"""
Model architecture: MobileNetV2 backbone (transfer learning) + custom head.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

import config


def build_model(num_classes: int = config.NUM_CLASSES, fine_tune: bool = False):
    """
    Build the weather classification model.

    Args:
        num_classes: number of output classes.
        fine_tune: if True, unfreeze the top backbone layers for fine-tuning.
    """
    base_model = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    if not fine_tune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Freeze everything before FINE_TUNE_AT_LAYER, unfreeze the rest.
        for layer in base_model.layers[: config.FINE_TUNE_AT_LAYER]:
            layer.trainable = False

    inputs = layers.Input(shape=config.INPUT_SHAPE)
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config.DENSE_UNITS_1, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_1)(x)
    x = layers.Dense(config.DENSE_UNITS_2, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="weather_classifier")
    return model, base_model
