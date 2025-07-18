import cv2
import numpy as np

def load_char_list(filepath):
    """Loads a character list from a file, one character per line (including space)."""
    print(f"Loading character list from: {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chars = [line.rstrip('\n') for line in f if line.strip('\n') != '']
        print(f"Loaded {len(chars)} characters from {filepath}.")
        return chars
    except FileNotFoundError:
        print(f"Error: Character list file not found at {filepath}. Returning empty list.")
        raise
    except Exception as e:
        print(f"Error loading character list from {filepath}: {e}. Returning empty list.")
        raise


def preprocess_image_for_ocr(image_path, target_size=(128, 32), binarize=False):
    """
    Preprocesses an image for OCR inference or training.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): (width, height) after resizing.
        binarize (bool): Apply Otsu's binarization.

    Returns:
        np.ndarray: Preprocessed image, shape (1, H, W), dtype float32.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or unable to read at {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if binarize:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)

    return img
