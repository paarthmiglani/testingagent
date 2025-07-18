# src/ocr/easyocr_infer.py
import easyocr
from PIL import Image
import numpy as np

# Initialize EasyOCR at module load (so it doesn't reload every time)
reader = easyocr.Reader(['en', 'hi', 'te'], gpu=False)  # add more languages as needed

def run_easyocr_on_image(image_file):
    """
    image_file: Streamlit UploadedFile or path.
    Returns: extracted text (str)
    """
    if hasattr(image_file, 'read'):
        # Streamlit UploadedFile: read as PIL Image
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
    else:
        img_np = image_file  # Already ndarray or path

    result = reader.readtext(img_np, detail=0)
    return "\n".join(result)
