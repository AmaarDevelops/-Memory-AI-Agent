import os
import time
import google.generativeai as genai
from google.api_core import exceptions

class GeminiEmbed:
    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name
        api_keys_str = os.getenv("GEMINI_API_KEY", "")
        self.api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
        self.current_key_index = 0

        if not self.api_keys:
            raise ValueError("No API Key found for Embedding.")

        self._configure()

    def _configure(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def embed_text(self, text: str):
        if not text or not text.strip():
            text = "empty_query"

        # Try every key available before giving up
        for attempt in range(len(self.api_keys)):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']

            except exceptions.ResourceExhausted:
                print(f"--- [EMBEDDER] Key {self.current_key_index + 1} quota hit. Rotating... ---")
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                self._configure()
                time.sleep(1) # Short breath
                continue

            except (exceptions.DeadlineExceeded, exceptions.ServiceUnavailable):
                print("--- [EMBEDDER] Server timed out. Retrying in 2s... ---")
                time.sleep(2)
                continue

        raise Exception("All embedding API keys are exhausted for today.")
