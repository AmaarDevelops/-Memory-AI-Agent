import os
import time
import google.generativeai as genai
from typing import List
from google.api_core import exceptions

class GeminiClient:
    """
    Gemini Client with built-in Key Rotation and Quota Handling.
    """

    def __init__(self, api_keys: List[str], model_name: str = 'gemini-1.5-flash', temperature: float = 0.3):
        self.api_keys = api_keys
        self.model_name = model_name
        self.temperature = temperature
        self.current_key_index = 0

        if not self.api_keys:
            raise ValueError('No API keys provided to GeminiClient.')

        # Initialize the first key
        self._configure_genai()


    def _configure_genai(self):
        """Internal method to switch the global genai config to the current key."""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature
            }
        )
        print(f"--- [SYSTEM] Active API Key Slot: {self.current_key_index + 1} ---")


    def _rotate_key(self):
        """Switches to the next available API key in the list."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_genai()


    def generate(self, system_prompt: str, user_prompt: str, memories: List[str] | None = None) -> str:
        """
        Generate response using LLM (Gemini) with automatic retry on quota exhaustion.
        """
        # Build the prompt
        full_prompt = system_prompt.strip()
        if memories:
            memory_block = "\n".join(f"- {m}" for m in memories)
            full_prompt += f"\n\nRelevant past memories: \n{memory_block}"

        full_prompt += f'\n\nUser: \n{user_prompt}\n\nAssistant:'

        # Try to generate, rotating keys if a 429 (ResourceExhausted) occurs
        for attempt in range(len(self.api_keys)):
            try:
                print(f"--- [LLM] Sending request to Gemini (Attempt {attempt + 1})... ---")
                response = self.model.generate_content(full_prompt)
                print("--- [LLM] Response received! ---")
                return response.text.strip()
            
            except exceptions.ResourceExhausted:
                print(f"--- [WARNING] Key slot {self.current_key_index + 1} exhausted (429). ---")

                if len(self.api_keys) > 1 and attempt < len(self.api_keys) - 1:
                    print("Rotating to backup key...")
                    self._rotate_key()
                    time.sleep(1)  # Brief pause to let the new key settle
                    continue
                else:
                    return "ERROR: All API keys have reached their quota limits for today."

            except Exception as e:
                return f"An unexpected error occurred in GeminiClient: {str(e)}"

        return "CRITICAL: Could not generate response after trying all keys."
