"""
Model loading and inference for the Smart Waste Sorting Streamlit app.
"""

import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image
from tensorflow import keras

from model.preprocess import preprocess_image

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "waste_classifier.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Recycling guidance shown alongside the prediction.
# طرق المعالجة/التدوير المقترحة لكل فئة
RECYCLING_GUIDE = {
    "cardboard": {
        "en": "Flatten and keep dry, then place in the paper/cardboard recycling bin. "
              "It is re-pulped into new paper products.",
        "ar": "افردها وحافظ عليها جافة، ثم ضعها في صندوق تدوير الورق/الكرتون. "
              "يُعاد تحويلها للب الورق لصناعة منتجات ورقية جديدة.",
    },
    "glass": {
        "en": "Rinse and place in the glass recycling bin (remove lids). "
              "Glass is crushed into cullet and re-melted — recyclable indefinitely.",
        "ar": "اغسلها وضعها في صندوق تدوير الزجاج (أزل الأغطية). "
              "يُكسر الزجاج ويُصهر من جديد، وهو قابل لإعادة التدوير بلا حدود.",
    },
    "metal": {
        "en": "Rinse cans and place in the metal recycling bin. "
              "Sorted magnetically/by eddy current, then melted into new alloys.",
        "ar": "اغسل العلب وضعها في صندوق تدوير المعادن. "
              "تُفرز بالمغناطيس أو بالتيار الدوامي ثم تُصهر لإنتاج سبائك جديدة.",
    },
    "paper": {
        "en": "Keep clean and dry, then recycle. Paper is pulped and reformed into new sheets.",
        "ar": "حافظ عليه نظيفًا وجافًا ثم أعد تدويره. يُنقع الورق ويُحول للب لصنع ورق جديد.",
    },
    "plastic": {
        "en": "Check the resin code, rinse, and recycle. Plastic is sorted by type, "
              "shredded, and reprocessed into pellets.",
        "ar": "تحقق من رمز نوع البلاستيك، اغسله، وأعد تدويره. يُفرز حسب النوع "
              "ثم يُجرش ويُعاد تصنيعه كحبيبات.",
    },
    "trash": {
        "en": "Not recyclable through standard streams — dispose of in general waste; "
              "some regions route this to waste-to-energy facilities.",
        "ar": "غير قابل لإعادة التدوير عبر المسارات القياسية — تخلص منه في المخلفات العامة، "
              "وقد يُحوَّل في بعض المناطق لمحطات توليد الطاقة من النفايات.",
    },
}


@lru_cache(maxsize=1)
def _load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Export the model from the notebook and place it in model/artifacts/."
        )
    model = keras.models.load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        index_to_label = {int(k): v for k, v in json.load(f).items()}

    return model, index_to_label


def inference(image: Image.Image) -> dict:
    """
    Run the full inference pipeline on a PIL image.

    Args:
        image: PIL.Image (from st.camera_input or st.file_uploader).

    Returns:
        dict with keys: label, confidence, probabilities, guidance
    """
    model, index_to_label = _load_model_and_labels()

    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    label = index_to_label[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {
        index_to_label[i]: float(p) for i, p in enumerate(probs)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "guidance": RECYCLING_GUIDE.get(label, {}),
    }
