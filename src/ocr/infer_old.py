# src/ocr/easyocr_infer.py

from paddleocr import PaddleOCR
import numpy as np
import cv2

# Initialize OCR (you can customize language as needed: 'en', 'hi', 'te', etc.)
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)  # For English
# ocr = PaddleOCR(use_angle_cls=True, lang='hi', show_log=False)  # For Hindi

def run_ocr_on_image(img_file):
    # img_file: can be path or file-like object (e.g., from Streamlit)
    if hasattr(img_file, 'read'):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif isinstance(img_file, str):
        img = cv2.imread(img_file)
    else:
        raise ValueError("Unsupported input to OCR")

    if img is None:
        return "Image could not be loaded!"

    # Run OCR
    result = ocr.ocr(img, cls=True)
    texts = [line[1][0] for line in result[0]]
    full_text = "\n".join(texts)
    return full_text if full_text else "No text detected!"
