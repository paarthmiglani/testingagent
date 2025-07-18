# Indic Heritage Chatbot

This repository contains code for an OCR pipeline and a simple NLP agent demo built with Streamlit.

## Components

- **src/ocr** – utilities for training and running OCR models using EasyOCR and other helpers.
- **src/nlp** – a minimal chat agent that can detect language, transliterate or translate text, and search CSV knowledge bases.
- **streamlit_app** – a small Streamlit interface combining the OCR pipeline and chat agent.

## Running the demo

1. Install dependencies (for example via `pip install -r requirements.txt`).
2. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
   ```
3. Upload an image or enter text to query the knowledge base.

The paths to transliteration tables and knowledge base CSVs in `streamlit_app/app.py` may need to be updated to match your local `data/` directory.
