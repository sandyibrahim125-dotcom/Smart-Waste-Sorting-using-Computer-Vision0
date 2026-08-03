from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    تجهيز الصورة قبل إدخالها للموديل
    """
    target_size = (224, 224) 
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
