"""
Builds Keras ImageDataGenerators for train / validation / test sets.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def get_data_generators():
    """Return (train_generator, val_generator, test_generator)."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    # No augmentation for val/test — only rescaling.
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator
