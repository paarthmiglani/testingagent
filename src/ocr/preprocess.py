import cv2
import numpy as np


def preprocess_image(img_path, size=(128, 32), binarize=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{img_path} not found")

    if binarize:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    return img
