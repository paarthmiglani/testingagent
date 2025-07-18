def guard_rails(user_input, ocr_result=None):
    # 1. Greetings
    if user_input.strip().lower() in ["hi", "hello", "namaste"]:
        return "Namaste! How can I help you with Indian heritage, texts, or culture today?"
    if user_input.strip().lower() in ["thanks", "thank you"]:
        return "You're welcome! If you have more questions, just ask."

    # 2. Help
    if "what can you do" in user_input.lower():
        return "You can ask about Indian scriptures, history, culture, ayurveda, or upload an image..."

    # 3. Empty input
    if not user_input.strip():
        return "Please type a question or upload an image to begin!"

    # 4. Non-heritage questions
    if any(x in user_input.lower() for x in ["ipl", "weather", "cricket", "stock"]):
        return "I specialize in Indian heritage, texts, and culture. Please ask about those topics!"

    # 5. Toxic/abusive (example, expand with a profanity list)
    if "badword" in user_input.lower():
        return "Let's keep our conversation respectful..."

    # 6. OCR image failure
    if ocr_result is not None and not ocr_result.strip():
        return "No readable text found in the image. Please try another image with clear text."

    # ... Add more as needed

    return None  # No guard rail triggered, proceed to normal search
