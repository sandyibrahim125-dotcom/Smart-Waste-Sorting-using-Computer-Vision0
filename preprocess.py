"""
Preprocessing utilities for the Smart Waste Sorting Streamlit app.

Keeps preprocessing identical to what was used at training time in the notebook
(MobileNetV2-style resize + preprocess_input), so predictions are consistent.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a model-ready batch of shape (1, 224, 224, 3).

    Args:
        image: PIL.Image opened from an uploaded file or camera input.

    Returns:
        np.ndarray ready to be passed to model.predict().
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.asarray(image).astype("float32")
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return array
