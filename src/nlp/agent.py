import glob
import os
from src.nlp.translator import IndicLanguageHandler
from src.nlp.search import keyword_search
from src.nlp.guardrails import guard_rails
class IndicCulturalChatAgent:
    def __init__(self, translit_paths, knowledge_files):
        self.lang_handler = IndicLanguageHandler(translit_paths)
        self.knowledge_files = knowledge_files  # dict: topic → csv_path

    def handle_input(self, user_input):
        lang = self.lang_handler.detect_lang(user_input)
        translit = self.lang_handler.transliterate_to_english(user_input, lang)
        eng_text = self.lang_handler.translate_to_english(translit, lang)

        # Search all knowledge bases
        found = []
        for topic, path in self.knowledge_files.items():
            results = keyword_search(eng_text, path)
            if not results.empty:
                found.append((topic, results))

        if found:
            # Return top result, or format as needed
            return found[0][1].to_dict('records')[0]
        else:
            return "Sorry, I could not find information on that."

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[2]
    translit_dir = BASE_DIR / "data" / "nlpdata"
    knowledge_dir = translit_dir
    agent = IndicCulturalChatAgent(translit_dir, knowledge_dir)
    print(agent.handle_input("धर्म"))
