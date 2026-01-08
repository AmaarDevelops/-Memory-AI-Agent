import requests
import json

class GeminiClient:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        print(f"--- [SYSTEM] Manual API Mode Active (Key {self.current_key_index + 1}) ---")

    def generate(self, system_prompt, user_prompt, memories=None):
        enhanced_system_prompt = (
            "You are a helpful AI Assistant for Smart Glasses. "
            "You have access to the user's past interactions provided in the Context section. "
            "ALWAYS use the Context to answer personal questions about the user (like codes, names, or preferences). "
            "If the information is in the Context, provide it directly."
        )

        # Clean up the memory text
        context_text = ""
        if memories:
            context_text = "\n".join([f"- {m}" for m in memories])
        else:
            context_text = "No relevant memories found for this specific query."

        full_prompt = f"{enhanced_system_prompt}\n\nCONTEXT FROM USER MEMORY:\n{context_text}\n\nUSER QUESTION: {user_prompt}"


        # 1. UPDATED URL: Using the stable v1 path
        # 2. UPDATED MODEL: Using gemini-2.5-flash-lite (Active for 2026 Free Tier)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={self.api_keys[self.current_key_index]}"

        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }

        try:
            print(f"--- [LLM] Requesting Gemini 2.5 Flash-Lite (Key {self.current_key_index + 1})... ---")
            response = requests.post(url, headers=headers, json=data)
            res_json = response.json()

            if response.status_code == 200:
                return res_json['candidates'][0]['content']['parts'][0]['text']
            else:
                # This will tell us if it's a quota issue (429) or something else
                error_msg = res_json.get('error', {}).get('message', 'Unknown Error')

                # If we hit the 2 RPM limit, rotate keys
                if "429" in str(response.status_code):
                    print("--- [QUOTA] Limit reached. Rotating key... ---")
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    return self.generate(system_prompt, user_prompt, memories)

                return f"API Error: {error_msg}"
        except Exception as e:
            print(f"Exception in Gemini Client :- {e}")
