# src/ocr/streamlit_helpers.py

import tempfile

def streamlit_image_to_path(uploaded_file, suffix=".jpg"):
    """Save a Streamlit UploadedFile to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
