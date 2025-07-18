import streamlit as st
import csv
from datetime import datetime
from src.nlp.agent import IndicCulturalChatAgent
from src.ocr.easyocr_infer import run_easyocr_on_image

# --- AGENT SETUP (adjust paths as needed) ---
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
translit_paths = {
    "hi": str(BASE_DIR / "data" / "hindi_transliterated.csv"),
    "te": str(BASE_DIR / "data" / "telugu_books.csv"),
    # add more if needed
}
knowledge_files = {
    "mahabharat": str(BASE_DIR / "data" / "Mahabharat.csv"),
    "gita": str(BASE_DIR / "data" / "MERGED_GITA_MANTRAS.csv"),
    "upanishads": str(BASE_DIR / "data" / "MERGED_upanishads.csv"),
    # ... add more
}
agent = IndicCulturalChatAgent(translit_paths, knowledge_files)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Indic Heritage Agent", page_icon="📚")
st.title("Indic Heritage Exploration Chatbot")

st.markdown("#### Ask about Indian texts, heritage, Ayurveda, Vedas, and more!")

input_mode = st.radio("Choose input type:", ["Text", "Image"])

if 'feedback_given' not in st.session_state:
    st.session_state['feedback_given'] = False
if 'last_answer' not in st.session_state:
    st.session_state['last_answer'] = None

user_input = None
ocr_text = None

if input_mode == "Text":
    user_input = st.text_area("Type your query here...", key="input_text")
elif input_mode == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        ocr_text = run_easyocr_on_image(img_file)
        st.write("Detected Text:", ocr_text)
        user_input = ocr_text

# --- AGENT RESPONSE & FEEDBACK ---
if user_input and user_input.strip():
    answer = agent.handle_input(user_input.strip())
    st.session_state['last_answer'] = answer
    st.session_state['last_query'] = user_input.strip()
    st.session_state['feedback_given'] = False

    st.markdown("### Agent's Answer")
    st.write(answer)

    # Feedback UI
    st.markdown("#### Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes", key="yes_feedback") and not st.session_state['feedback_given']:
            with open("feedback_log.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), st.session_state['last_query'], st.session_state['last_answer'], "positive"])
            st.success("Thanks for your feedback!")
            st.session_state['feedback_given'] = True
    with col2:
        if st.button("👎 No", key="no_feedback") and not st.session_state['feedback_given']:
            with open("feedback_log.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), st.session_state['last_query'], st.session_state['last_answer'], "negative"])
            st.warning("Thanks for your feedback! We'll try to improve.")
            st.session_state['feedback_given'] = True

st.markdown("---")
st.markdown("Powered by EasyOCR & Indic NLP (demo)")
